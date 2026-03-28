#pragma once
#include <atomic>
#include <new>
#include <thread>
#include <iostream>
#include <vector>

// Align the node to the cache line size to prevent false sharing between threads
struct alignas(std::hardware_destructive_interference_size) MCSNode {
    std::atomic<MCSNode*> next{nullptr};
    std::atomic<bool> locked{false};
};

class MCSLock {
    std::atomic<MCSNode*> tail{nullptr};

public:
    // A thread MUST pass in its own local node to acquire the lock.
    void lock(MCSNode* node) {
        node->next.store(nullptr, std::memory_order_relaxed);
        node->locked.store(true, std::memory_order_relaxed);

        // Swap our node into the tail of the queue
        MCSNode* prev = tail.exchange(node, std::memory_order_acq_rel);
        
        if (prev != nullptr) {
            // Someone else has the lock. Link our node to theirs.
            prev->next.store(node, std::memory_order_release);
            
            // C++20: Wait efficiently rather than burning CPU cycles in a pure spin loop
            node->locked.wait(true, std::memory_order_acquire);
        }
    }

    void unlock(MCSNode* node) {
        MCSNode* next = node->next.load(std::memory_order_acquire);
        
        if (next == nullptr) {
            // We think we are the last node in the queue.
            MCSNode* expected = node;
            
            // Try to set the tail back to nullptr.
            if (tail.compare_exchange_strong(expected, nullptr, std::memory_order_release, std::memory_order_relaxed)) {
                return; // Successfully unlocked, queue is empty.
            }
            
            // If compare_exchange failed, another thread is actively running lock() 
            // but hasn't linked its node to ours yet. Spin until they do.
            while ((next = node->next.load(std::memory_order_acquire)) == nullptr) {
                std::this_thread::yield(); 
            }
        }

        // Pass the lock to the next thread
        next->locked.store(false, std::memory_order_release);
        
        // C++20: Wake up the next thread that is waiting on its local flag
        next->locked.notify_one();
    }
};