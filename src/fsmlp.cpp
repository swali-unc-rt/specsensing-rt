#include "fsmlp.hpp"

#include <atomic>

#include <litmus/ctrlpage.h>
#include <litmus.h>

#include <immintrin.h>

//#include <libsmctrl.h>

using std::atomic;

// USMLP fsmlp(GPC_ALL_BUT_GPC6, 1000);
//USMLP smlp(GPC6, RT_NUM_CPUS );
//USMLP all_smlp(GPC0 | GPC1 | GPC2 | GPC3 | GPC4 | GPC5 | GPC6, RT_NUM_CPUS );

void FSMLP::processQueues() {
    MCSNode node;
    fsmlp_lock.lock(&node);
    
    // First, check the status of SQ streams for completion
    //for( auto it = sq.begin(); it != sq.end(); ) {

    auto cpos = sqstart;
    while( cpos != sqend ) {
        //GPURequest* req = *it;
        GPURequest* req = SQ[cpos];
        if( !req ) {
            cpos = (cpos+1) % QSIZEMAX;
            continue;
        }

        cudaError_t status = cudaStreamQuery( *(req->stream) );
        if( status == cudaSuccess ) {
            // Completed
            mask |= req->smctrl_mask_assigned;
            //it = sq.erase(it);
            SQ[cpos] = nullptr;
            if( cpos == sqstart ) {
                sqstart = (sqstart + 1) % QSIZEMAX;
                cpos = sqstart;
            } else
                cpos = (cpos+1) % QSIZEMAX;
            //req->job_barrier->arrive_and_drop();
            auto arrival_token = req->job_barrier.arrive();
            //printf("recover mask 0x%lx from stream\n", req->smctrl_mask_assigned);
        } else if( status == cudaErrorNotReady ) {
            // Still running
            ++cpos;
        } else {
            // Some other error
            fprintf(stderr,"Error querying CUDA stream\n");
            ++cpos;
        }
    }

    // Do we have any available SMs?
    //while( !fq.empty() && mask && !stopped ) {
    while( fqstart != fqend && mask && !stopped ) {
        GPURequest* req;
        //req = fq.front();
        req = FQ[fqstart];
        //fq.pop_front();
        fqstart = (fqstart + 1) % QSIZEMAX;

        // Anything in the PQ?
        if( pq.empty() )
            fqsize--;
        else {
            // Move the top from PQ to the back of FQ
            GPURequest* next_req = pq.top();
            pq.pop();
            //fq.push_back( next_req );
            FQ[fqend] = next_req;
            fqend = (fqend + 1) % QSIZEMAX;
        }

        // Assign SMs
        int tpcsToUse = __builtin_popcountll( mask );

        // Can this request take this many TPCs?
        while( !(req->smctrl_tpcs_allowed & (1ULL << (tpcsToUse-1))) )
            tpcsToUse--;
        req->smctrl_mask_assigned = 0;
        for( int i = 0; i < tpcsToUse; ++i ) {
            int tpc = __builtin_ffsll(mask) - 1;
            req->smctrl_mask_assigned |= (1ULL << tpc);
            mask &= ~(1ULL << tpc);
        }

        // Set the stream mask to the opposite of this
        // because libsmctrl sets 1 as a disabled TPC
        printf("assigning mask 0x%lx to stream\n", req->smctrl_mask_assigned);
        libsmctrl_set_stream_mask( *req->stream, ~req->smctrl_mask_assigned );

        // Do the launch
        req->gpu_launch_fn( req );

        // Add to SQ, and lets see if we can handle more
        //sq.push_front( req );
        SQ[sqend] = req;
        sqend = (sqend + 1) % QSIZEMAX;
    }
    //exit_fsmlp_lock();
    fsmlp_lock.unlock(&node);
}

lt_t FSMLP::submitRequest(//uint64_t smctrl_tpcs_allowed,
    //cudaStream_t* stream,
    //std::function<void()> gpu_launch_fn) {
    GPURequest* req ) {

    //auto startclock = litmus_clock();
    //lt_t endclock;
    
    // struct control_page* cpage = get_ctrl_page();
    // if( !cpage ) {
    //     fprintf(stderr,"FSMLP: submitRequest could not get control page\n");
    //     return 0;
    // }
    if( !req->stream ) {
        fprintf(stderr,"FSMLP: submitRequest called with null stream\n");
        return 0;
    }
    if( !(req->smctrl_tpcs_allowed & 1) ) {
        fprintf(stderr,"FSMLP: submitRequest called with invalid tpc allowance mask 0x%lx\n", req->smctrl_tpcs_allowed);
        return 0;
    }

    /*GPURequest* req = new GPURequest{
        .abs_deadline = cpage->deadline,
        .smctrl_tpcs_allowed = smctrl_tpcs_allowed,
        .smctrl_mask_assigned = 0,
        //.stream = stream,
        .gpu_launch_fn = gpu_launch_fn,
        .job_barrier = new std::barrier(2)
    };*/

    // A quick enqueue operation
    //get_fsmlp_lock();
    fsmlp_lock.lock(&req->lock_node);
    if( mask ) {
        // can immediately satisfy
        //sq.push_front( req );
        SQ[sqend] = req;
        sqend = (sqend + 1) % QSIZEMAX;

        // Assign SMs
        int tpcsToUse = __builtin_popcountll( mask );
        // Can this request take this many TPCs?
        while( !(req->smctrl_tpcs_allowed & (1ULL << (tpcsToUse-1))) )
            tpcsToUse--;
        if( !tpcsToUse ) {
            fprintf(stderr,"FSMLP: submitRequest could not assign any TPCs despite mask being non-zero\n");
            tpcsToUse = 1;
        }
        req->smctrl_mask_assigned = 0;
        for( int i = 0; i < tpcsToUse; ++i ) {
            int tpc = __builtin_ffsll(mask) - 1;
            req->smctrl_mask_assigned |= (1ULL << tpc);
            mask &= ~(1ULL << tpc);
        }

        // Set the stream mask to the opposite of this
        // because libsmctrl sets 1 as a disabled TPC
        printf("assigning mask 0x%lx to stream\n", req->smctrl_mask_assigned);
        libsmctrl_set_stream_mask( *req->stream, ~req->smctrl_mask_assigned );
        // Do the launch
        //endclock = litmus_clock();
        req->gpu_launch_fn( req );
        //exit_fsmlp_lock();
        fsmlp_lock.unlock(&req->lock_node);
    } else if( fqsize < m ) {
        // Place in FQ
        //fq.push_back( req );
        FQ[fqend] = req;
        fqend = (fqend + 1) % QSIZEMAX;
        fqsize++;
        //exit_fsmlp_lock();
        fsmlp_lock.unlock(&req->lock_node);
        //endclock = litmus_clock();
        //printf("rq in fq\n");
    } else {
        // Place in PQ
        pq.push( req );
        //exit_fsmlp_lock();
        fsmlp_lock.unlock(&req->lock_node);
        //endclock = litmus_clock();
        //printf("rq in pq\n");
    }

    // The GPU task will handle notifying when to proceed
    req->job_barrier.arrive_and_wait();
    //printf("-=-=-=-=-=-=-=-=-=- fsmlp finish\n");

    // Now do cleanup
    //delete req->job_barrier;
    //delete req;

    //return endclock - startclock;
    return 0;
}

bool cmpGPURequest(GPURequest* a, GPURequest* b) {
    return a->abs_deadline > b->abs_deadline;
}

void FSMLP::stop() {
    stopped = true;

    while( fqstart != fqend ) {
        auto f = FQ[fqstart];
        auto arrival_token = f->job_barrier.arrive();
        fqstart = (fqstart + 1) % QSIZEMAX;
    }
    while( !pq.empty() ) {
        auto p = pq.top();
        auto arrival_token = p->job_barrier.arrive();
        pq.pop();
    }

    while( sqstart != sqend ) {
        auto s = SQ[sqstart];
        if( s ) {
            auto arrival_token = s->job_barrier.arrive();
        }
        sqstart = (sqstart + 1) % QSIZEMAX;
    }
}