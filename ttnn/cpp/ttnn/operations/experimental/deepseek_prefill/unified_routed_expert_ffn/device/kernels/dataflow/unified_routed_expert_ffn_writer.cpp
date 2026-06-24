// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Single responsibility: pop `cb_out` (the down matmul's per-core final
// block, packed one subblock at a time) and write tiles to the DRAM-
// interleaved output tensor at this core's (mt, nt_d) tile region, looped
// over `effective_chunks` chunks. Reads the device-side counts/idx
// scratch CBs to compute effective_chunks. The activated tiles are
// distributed across the M-row by the READER via L1 multicast — there is
// no DRAM scratch round-trip or cross-core barrier.
//
// Output placement has two modes (selected by the `direct_write` CT flag):
//   * direct_write == 0: writes start at tile row 0. The FFN op writes to
//     a per-expert output tensor; a separate ttnn::insert handles placement
//     into any shared destination buffer.
//   * direct_write == 1: this expert's output is written directly into a
//     shared destination buffer at the expert's region offset. The kernel
//     reads start[global_expert_id] from `start` (= expert_region_offsets)
//     device-side and adds (start / TILE_HEIGHT) tile-rows to every output
//     row — fusing what ttnn::insert would otherwise do as a separate op
//     (no temp-buffer DRAM round-trip).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);
    // start (= expert_region_offsets) address. Only read when direct_write.
    const uint32_t start_addr = get_arg_val<uint32_t>(3);
    // Leader-completion sync trailing args (see program factory). Consumed only
    // at the very end of the kernel, after all output writes for this expert.
    const uint32_t is_leader = get_arg_val<uint32_t>(4);
    const uint32_t leader_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t leader_noc_y = get_arg_val<uint32_t>(6);
    const uint32_t leader_sync_sem_id = get_arg_val<uint32_t>(7);
    const uint32_t num_non_leader = get_arg_val<uint32_t>(8);
    // Global-sem mcast args (only used on the leader path).
    const uint32_t has_global_sem = get_arg_val<uint32_t>(9);
    const uint32_t global_sem_mcast_start_x = get_arg_val<uint32_t>(10);
    const uint32_t global_sem_mcast_start_y = get_arg_val<uint32_t>(11);
    const uint32_t global_sem_mcast_end_x = get_arg_val<uint32_t>(12);
    const uint32_t global_sem_mcast_end_y = get_arg_val<uint32_t>(13);
    const uint32_t global_sem_num_dests = get_arg_val<uint32_t>(14);
    const uint32_t global_sem_l1_addr = get_arg_val<uint32_t>(15);

    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(4);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(10);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(12);
    // NEW: device-side count read.
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(13);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(14);
    constexpr uint32_t local_expert_id = get_compile_time_arg_val(15);
    // M_tiles_full: total tile-row count of the FFN *input* (x). When the
    // kernel runs more chunks than strictly needed (because
    // M_tiles_full % chunk_M_tiles != 0), the last chunk has writer
    // destinations past M_tiles_full — we skip those source rows here.
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(16);
    // direct_write: 0 -> write to per-expert output at row 0 (standalone FFN).
    //               1 -> write into shared buffer at start[global_id]/TILE.
    constexpr uint32_t direct_write = get_compile_time_arg_val(17);
    // dst_M_tiles: tile-row count of the *destination* buffer. Equals
    // M_tiles_full in non-direct mode; the shared buffer's row count in
    // direct-write mode (used to bound destination writes).
    constexpr uint32_t dst_M_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t cb_start_scratch = get_compile_time_arg_val(19);

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;

    constexpr uint32_t out_accessor_offset = 20;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    constexpr uint32_t start_accessor_offset = out_args.next_compile_time_args_offset();
    constexpr auto start_args = TensorAccessorArgs<start_accessor_offset>();
    const auto start_acc = TensorAccessor(start_args, start_addr);

    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    // Wait for the reader's counts/idx push and compute effective_chunks =
    // ceil(count / chunk_M_tiles). The writer drains cb_out per chunk;
    // bounding the loop here is required because the reader and compute
    // bound theirs too — without this, the writer would wait forever on
    // cb_out for chunks the compute never pushes.
    cb_wait_front(cb_counts_scratch, 1);
    cb_wait_front(cb_idx_scratch, 1);
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_counts_scratch));
    const uint32_t idx_l1 = get_read_ptr(cb_idx_scratch);
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1);
    const uint32_t global_expert_id = idx_ptr[local_expert_id];
    const uint32_t count_value = counts_ptr[global_expert_id];
    const uint32_t count_tiles = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t effective_chunks_runtime = (count_tiles + chunk_M_tiles - 1) / chunk_M_tiles;
    const uint32_t effective_chunks = effective_chunks_runtime < num_chunks ? effective_chunks_runtime : num_chunks;

    // Destination tile-row offset for direct-write mode. In direct-write
    // mode the output buffer is a SHARED buffer and this expert's slice
    // begins at start[global_expert_id] (token row); convert to tile rows.
    // Mirrors ttnn::insert's writer: start_tile_row = start_value / TILE.
    uint32_t row_offset_tiles = 0;
    if constexpr (direct_write != 0) {
        const uint32_t start_l1 = get_write_ptr(cb_start_scratch);
        noc_async_read_page(0, start_acc, start_l1);
        noc_async_read_barrier();
        const volatile tt_l1_ptr uint32_t* start_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(start_l1);
        const uint32_t start_value = start_ptr[global_expert_id];
        row_offset_tiles = start_value / TILE_HEIGHT;
    }

    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        const uint32_t row0 = chunk * chunk_M_tiles + my_mt * per_core_M;
        const uint32_t col0 = my_nt_d * per_core_N_d;
        for (uint32_t sb_m = 0; sb_m < d_in1_num_subblocks_M; ++sb_m) {
            for (uint32_t sb_n = 0; sb_n < d_in1_num_subblocks_N; ++sb_n) {
                cb_wait_front(cb_out, d_out_subblock_num_tiles);
                uint32_t l1_read = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < d_out_subblock_h; ++i) {
                    for (uint32_t j = 0; j < d_out_subblock_w; ++j) {
                        const uint32_t row = row0 + sb_m * d_out_subblock_h + i;
                        const uint32_t col = col0 + sb_n * d_out_subblock_w + j;
                        // `row` indexes the FFN *input* (x) tile-rows; the
                        // destination tile-row adds the per-expert region
                        // offset (0 in non-direct mode).
                        const uint32_t dst_row = row_offset_tiles + row;
                        // Bounds that decide whether this is a real output tile
                        // for this expert:
                        //   * col < N_down_tiles_full: GRID_X=11 ceil_div
                        //     produces phantom output cols past actual N.
                        //   * row < M_tiles_full: ceil_div of M produces a
                        //     last-chunk tail past actual M when
                        //     M_tiles_full doesn't divide chunk_M_tiles.
                        //   * row < count_tiles: the last chunk's per_core_M
                        //     rows extend past count_tiles when count_tiles
                        //     is not chunk-aligned.
                        if (col < N_down_tiles_full && row < M_tiles_full && row < count_tiles) {
                            // The destination tile-row must stay inside the
                            // (possibly shared) output buffer. ttnn::insert
                            // asserted the whole-slice fit
                            // (start_tile_idx + num_tiles <= global_num_tiles);
                            // assert the per-tile equivalent so an over-capacity
                            // region offset fails loudly in watcher builds. The
                            // guard below keeps Release builds safe (skip the OOB
                            // write rather than corrupt DRAM, since ASSERT is a
                            // no-op there).
                            ASSERT(dst_row < dst_M_tiles);
                            if (dst_row < dst_M_tiles) {
                                const uint32_t tile_idx = dst_row * N_down_tiles_full + col;
                                noc_async_write_page(tile_idx, out_acc, l1_read);
                            }
                        }
                        l1_read += out_tile_bytes;
                    }
                }
                // Wait for the writes to LEAVE this core (departed sender);
                // doesn't wait for the DRAM round-trip. Safe to reuse the L1
                // slot now — the NoC has captured the data. ~10x faster than
                // noc_async_write_barrier per subblock at small per_core_M.
                noc_async_writes_flushed();
                cb_pop_front(cb_out, d_out_subblock_num_tiles);
            }
        }
    }
    // Ensure all outstanding writes complete at the destination before the
    // kernel returns (the next dispatched op may read this output).
    noc_async_write_barrier();

    // Leader-completion sync. Every core reaches here only after its own output
    // writes have landed (the barrier above). Non-leader cores signal the leader
    // core's leader_sync sem; the leader waits for all num_non_leader signals so
    // it is provably the last core to finish, then multicast-increments the
    // global semaphore (+1 for this expert).
    const uint32_t leader_sem_l1 = get_semaphore(leader_sync_sem_id);
    volatile tt_l1_ptr uint32_t* leader_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(leader_sem_l1);
    if (is_leader == 0) {
        // Non-leader: increment the leader core's copy of the sync sem.
        const uint64_t leader_noc_addr = get_noc_addr(leader_noc_x, leader_noc_y, leader_sem_l1);
        noc_semaphore_inc(leader_noc_addr, 1);
        noc_async_atomic_barrier();
    } else {
        // Leader: block until every non-leader has signaled completion.
        if (num_non_leader > 0) {
            noc_semaphore_wait(leader_sem_ptr, num_non_leader);
        }
        // Then bump every instance of the global semaphore.
        if (has_global_sem) {
            // This writer kernel runs on the DRAM-write NOC (NOC1 on Blackhole),
            // whose coordinate system is inverted vs NOC0: get_noc_multicast_addr
            // maps each coord x -> (size-1-x). NOC_MULTICAST_ADDR requires the
            // rectangle's start <= end in the *target NOC's* hardware coords, but
            // the host computed start/end as min/max in NOC0/virtual coords. On
            // NOC1 the inversion flips that ordering, so we must pass the corners
            // as (max -> min); on NOC0 we pass them as-is (min -> max).
            uint32_t mc_sx = global_sem_mcast_start_x;
            uint32_t mc_sy = global_sem_mcast_start_y;
            uint32_t mc_ex = global_sem_mcast_end_x;
            uint32_t mc_ey = global_sem_mcast_end_y;
            if (noc_index == 1) {
                mc_sx = global_sem_mcast_end_x;
                mc_sy = global_sem_mcast_end_y;
                mc_ex = global_sem_mcast_start_x;
                mc_ey = global_sem_mcast_start_y;
            }
            const uint64_t gsem_mcast_noc_addr = get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, global_sem_l1_addr);
            noc_semaphore_inc_multicast(gsem_mcast_noc_addr, 1, global_sem_num_dests);
            noc_async_atomic_barrier();
        }
    }
}
