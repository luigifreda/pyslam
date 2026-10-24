/*
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PYSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/messages.h"

namespace py = pybind11;
// Wrapper functions to match Python signatures
namespace pyslam {

// Wrapper for the g2o.Flag to be used in the optimizer_g2o.cpp file
// The g2o.Flag is a Python object that is used to abort the optimization: We cannot access
// directly to the pointer of its stored boolean value (readonly). We wrap it with this class to
// and use a thread to sync the boolean value with the Python object.
class PyG2oAbortFlag {
    constexpr static int kMonitorIntervalMs = 5; // milliseconds

  private:
    py::object abort_flag_obj;
    bool abort_value;
    std::atomic<bool> should_stop_monitoring;
    std::thread monitor_thread;

  public:
    // Disable copy and move constructors and assignment operators
    PyG2oAbortFlag(const PyG2oAbortFlag &) = delete;
    PyG2oAbortFlag &operator=(const PyG2oAbortFlag &) = delete;
    PyG2oAbortFlag(PyG2oAbortFlag &&) = delete;
    PyG2oAbortFlag &operator=(PyG2oAbortFlag &&) = delete;

    PyG2oAbortFlag(py::object obj)
        : abort_flag_obj(obj), abort_value(false), should_stop_monitoring(false) {
        if (!obj.is_none()) {
            // Initialize the boolean value with the Python object
            try {
                abort_value = obj.attr("value").cast<bool>();
            } catch (const std::exception &e) {
                MSG_ERROR_STREAM("Error initializing abort flag: " << e.what());
                abort_value = false;
            }
            // Start monitoring thread to sync the boolean value with the Python object
            monitor_thread = std::thread(&PyG2oAbortFlag::monitor_flag, this);
        }
    }

    ~PyG2oAbortFlag() { stop_monitoring(); }

    void stop_monitoring() {
        if (monitor_thread.joinable()) {
            should_stop_monitoring = true;
            monitor_thread.join();
        }
    }

    bool *get_value_ptr() { return abort_flag_obj.is_none() ? nullptr : &abort_value; }

    bool get_value() { return abort_value; }

  private:
    void monitor_flag() {
        // Monitor the boolean value of the g2o.Flag python object and update the local boolean
        // value. We must hold the GIL each time we touch the Python object to avoid corrupting the
        // interpreter state when the monitoring thread runs in the background.
        while (!should_stop_monitoring) {
            try {
                py::gil_scoped_acquire acquire;
                if (!abort_flag_obj.is_none()) {
                    abort_value = abort_flag_obj.attr("value").cast<bool>();
                }
            } catch (const std::exception &e) {
                // Handle any Python exceptions (e.g. interpreter shutdown)
                MSG_ERROR_STREAM("Error monitoring abort flag: " << e.what());
            }
            // Sleep for a short interval (e.g., 1ms)
            std::this_thread::sleep_for(std::chrono::milliseconds(kMonitorIntervalMs));
        }
    }
};

// Wrapper for the Python lock to be used in the optimizer_g2o.cpp file
// It wraps a Python lock object and provides a RAII lock mechanism.
class PyLock {
  private:
    py::object lock_obj;
    bool is_acquired;

  public:
    PyLock(py::object lock) : lock_obj(lock), is_acquired(false) {}

    void lock() {
        if (!lock_obj.is_none()) {
            py::gil_scoped_acquire acquire;
            lock_obj.attr("acquire")();
            is_acquired = true;
        }
    }

    void unlock() {
        if (!lock_obj.is_none() && is_acquired) {
            py::gil_scoped_acquire acquire;
            lock_obj.attr("release")();
            is_acquired = false;
        }
    }

    ~PyLock() {
        if (is_acquired) {
            unlock();
        }
    }
};

// Template-based RAII lock wrapper that works with both std::mutex and PyLock
template <typename LockType> class PyLockGuard {
  private:
    LockType *lock_ptr;
    bool is_locked;

  public:
    // Constructor - acquires the lock
    explicit PyLockGuard(LockType *lock) : lock_ptr(lock), is_locked(false) {
        if (lock_ptr) {
            lock_ptr->lock();
            is_locked = true;
        }
    }

    // Destructor - releases the lock
    ~PyLockGuard() {
        if (lock_ptr && is_locked) {
            lock_ptr->unlock();
        }
    }

    // Disable copy constructor and assignment operator
    PyLockGuard(const PyLockGuard &) = delete;
    PyLockGuard &operator=(const PyLockGuard &) = delete;

    // Allow move constructor for potential future use
    PyLockGuard(PyLockGuard &&other) noexcept
        : lock_ptr(other.lock_ptr), is_locked(other.is_locked) {
        other.lock_ptr = nullptr;
        other.is_locked = false;
    }

    PyLockGuard &operator=(PyLockGuard &&other) noexcept {
        if (this != &other) {
            if (lock_ptr && is_locked) {
                lock_ptr->unlock();
            }
            lock_ptr = other.lock_ptr;
            is_locked = other.is_locked;
            other.lock_ptr = nullptr;
            other.is_locked = false;
        }
        return *this;
    }

    // Check if lock is held
    bool is_held() const { return is_locked; }
};

// Wrapper class for std::mutex to make it work as a Python context manager
// This class is registered in the cpp_core_module.cpp file via the bind_mutex_wrapper() function
// in the file mutex_helper.h
template <typename MutexType> class PyMutexWrapperT {
  public:
    explicit PyMutexWrapperT(MutexType &m) : mutex_(m) {}

    PyMutexWrapperT(const PyMutexWrapperT &) = delete;
    PyMutexWrapperT &operator=(const PyMutexWrapperT &) = delete;
    PyMutexWrapperT(PyMutexWrapperT &&) = delete;
    PyMutexWrapperT &operator=(PyMutexWrapperT &&) = delete;

    void lock() {
        py::gil_scoped_release nogil;
        mutex_.lock();
        is_locked_.store(true, std::memory_order_release);
    }
    void unlock() {
        is_locked_.store(false, std::memory_order_release);
        mutex_.unlock();
    }

    bool acquire(bool blocking = true, double timeout = -1) {
        if (!blocking) {
            py::gil_scoped_release nogil;
            if (mutex_.try_lock()) {
                is_locked_.store(true, std::memory_order_release);
                return true;
            }
            return false;
        }
        if (timeout < 0) {
            py::gil_scoped_release nogil;
            mutex_.lock();
            is_locked_.store(true, std::memory_order_release);
            return true;
        }
        if constexpr (std::is_same_v<MutexType, std::timed_mutex> ||
                      std::is_same_v<MutexType, std::recursive_timed_mutex>) {
            py::gil_scoped_release nogil;
            bool ok = mutex_.try_lock_for(std::chrono::duration<double>(timeout));
            if (ok)
                is_locked_.store(true, std::memory_order_release);
            return ok;
        } else {
            // Implement timeout with polling
            py::gil_scoped_release nogil;
            auto start = std::chrono::steady_clock::now();
            while (true) {
                if (mutex_.try_lock()) {
                    is_locked_.store(true, std::memory_order_release);
                    return true;
                }
                const auto elapsed = std::chrono::steady_clock::now() - start;
                if (elapsed >= std::chrono::duration<double>(timeout)) {
                    is_locked_.store(false, std::memory_order_release);
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void release() { unlock(); }

    bool locked() { return is_locked_.load(std::memory_order_acquire); }

    PyMutexWrapperT &__enter__() {
        py::gil_scoped_release nogil;
        mutex_.lock();
        is_locked_.store(true, std::memory_order_release);
        return *this;
    }

    bool __exit__(py::object, py::object, py::object) {
        is_locked_.store(false, std::memory_order_release);
        mutex_.unlock();
        return false; // don’t suppress exceptions
    }

  private:
    MutexType &mutex_;
    std::atomic<bool> is_locked_{false};
};

using PyMutexWrapper = PyMutexWrapperT<std::mutex>;
using PyRMutexWrapper = PyMutexWrapperT<std::recursive_mutex>;
using PyTMutexWrapper = PyMutexWrapperT<std::timed_mutex>;
using PyTRMutexWrapper = PyMutexWrapperT<std::recursive_timed_mutex>;

// ------------------------------------------------------------
// VectorProxy: Templated gateway wrapper for std::vector containers
// Provides Python list-like interface with zero-copy access
// ------------------------------------------------------------

template <typename T> class VectorProxy {
  public:
    // Constructor - borrows a reference to the std::vector
    explicit VectorProxy(std::vector<T> &vec) : vec_(&vec) {}

    // Constructor - owns a standalone vector (used for pickled instances)
    explicit VectorProxy(std::shared_ptr<std::vector<T>> vec)
        : vec_(vec.get()), owner_(std::move(vec)) {}

    // Get item at index
    T &getitem(size_t idx) {
        if (idx >= vec_->size()) {
            throw py::index_error("VectorProxy index out of range");
        }
        return (*vec_)[idx];
    }

    // Set item at index
    void setitem(size_t idx, const T &value) {
        if (idx >= vec_->size()) {
            throw py::index_error("VectorProxy index out of range");
        }
        (*vec_)[idx] = value;
    }

    // Append an item (zero-copy, directly modifies the C++ vector)
    void append(const T &value) { vec_->push_back(value); }

    // Extend with multiple items
    void extend(py::iterable items) {
        for (auto item : items) {
            vec_->push_back(py::cast<T>(item));
        }
    }

    // Get size
    size_t size() const { return vec_->size(); }

    // Check if empty
    bool empty() const { return vec_->size() == 0; }

    // Clear the vector
    void clear() { vec_->clear(); }

    // Convert to Python list (creates a copy)
    py::list to_list() {
        py::list result;
        for (const auto &item : *vec_) {
            result.append(py::cast(item));
        }
        return result;
    }

    // Iterator support
    class Iterator {
      public:
        Iterator(std::vector<T> &vec, size_t idx) : vec_(vec), idx_(idx) {}
        T &operator*() { return vec_[idx_]; }
        Iterator &operator++() {
            ++idx_;
            return *this;
        }
        bool operator!=(const Iterator &other) const { return idx_ != other.idx_; }

      private:
        std::vector<T> &vec_;
        size_t idx_;
    };

    Iterator begin() { return Iterator(*vec_, 0); }
    Iterator end() { return Iterator(*vec_, vec_->size()); }

    // Get reference to underlying vector (for advanced use cases)
    std::vector<T> &get_vector() { return *vec_; }
    const std::vector<T> &get_vector() const { return *vec_; }

    // Pickling support: serialize as a Python list
    // This allows VectorProxy to be pickled when passed through multiprocessing queues
    py::tuple __getstate__() const {
        py::list items;
        for (const auto &item : *vec_) {
            items.append(py::cast(item));
        }
        return py::make_tuple(items);
    }

  private:
    std::vector<T> *vec_;                   // Non-owning pointer to the vector
    std::shared_ptr<std::vector<T>> owner_; // Optional owner for standalone vectors
};

} // namespace pyslam