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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace pyslam {

// OrderedDict is a dictionary that maintains insertion order
// ------------------------------------------------------------------------------------------
// Operations	                   | Time Complexity   | Notes
// ------------------------------------------------------------------------------------------
// Lookup (find, at, operator[])   | O(1) average	   | Uses hash map
// Insertion (emplace, try_emplace)| O(1) average	   | Hash map + list append
// Erase by key	                   | O(1) average	   | Hash map + list erase
// Erase by iterator	           | O(1)	           | Direct list iterator
// Iteration	                   | O(n)	           | Sequential list traversal
// Size	                           | O(1)	           | Hash map size
// ------------------------------------------------------------------------------------------
// Space Complexity:
// ------------------------------------------------------------------------------------------
// - O(n) : Each element stored once in list + hash map entry
// - Memory overhead : ~2x compared to simple std::unordered_map (due to dual storage)
// ------------------------------------------------------------------------------------------

template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
          class Allocator = std::allocator<std::pair<Key, T>>>
class OrderedDict {
    struct node {
        Key key;
        T value;

        node(Key k, T v) : key(std::move(k)), value(std::move(v)) {}
        node(const node &) = default;
        node(node &&) noexcept(std::is_nothrow_move_constructible<Key>::value &&
                               std::is_nothrow_move_constructible<T>::value) = default;
        node &operator=(const node &) = default;
        node &operator=(node &&) noexcept(std::is_nothrow_move_assignable<Key>::value &&
                                          std::is_nothrow_move_assignable<T>::value) = default;
    };

    using list_alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<node>;
    using list_type = std::list<node, list_alloc>;
    using list_iter = typename list_type::iterator;
    using list_citer = typename list_type::const_iterator;

    using map_alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<
        std::pair<const Key, list_iter>>;
    using index_type = std::unordered_map<Key, list_iter, Hash, KeyEqual, map_alloc>;

  public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const Key, T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using allocator_type = Allocator;

    // Expose references like std::map (const key, mutable value)
    using reference = std::pair<const Key &, T &>;
    using const_reference = std::pair<const Key &, const T &>;

  private:
    // Proxy that materializes a pair<const Key, T> and returns its address.
    struct arrow_proxy {
        value_type p;
        arrow_proxy(const Key &k, T &v) : p(k, v) {}
        const value_type *operator->() const { return &p; }
        value_type *operator->() { return &p; }
    };
    struct const_arrow_proxy {
        value_type p; // value_type is pair<const Key, T>
        const_arrow_proxy(const Key &k, const T &v) : p(k, v) {}
        const value_type *operator->() const { return &p; }
    };

  public:
    class iterator {
        list_iter it_{};

      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = OrderedDict::value_type;
        using difference_type = std::ptrdiff_t;
        using reference = OrderedDict::reference;
        using pointer = void;

        iterator() = default;
        explicit iterator(list_iter it) : it_(it) {}

        reference operator*() const { return reference{it_->key, it_->value}; }
        arrow_proxy operator->() const { return arrow_proxy{it_->key, it_->value}; }

        iterator &operator++() {
            ++it_;
            return *this;
        }
        iterator operator++(int) {
            iterator tmp = *this;
            ++*this;
            return tmp;
        }
        iterator &operator--() {
            --it_;
            return *this;
        }
        iterator operator--(int) {
            iterator tmp = *this;
            --*this;
            return tmp;
        }

        friend bool operator==(const iterator &a, const iterator &b) { return a.it_ == b.it_; }
        friend bool operator!=(const iterator &a, const iterator &b) { return !(a == b); }

        list_iter base() const { return it_; }
    };

    class const_iterator {
        list_citer it_{};

      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = OrderedDict::value_type;
        using difference_type = std::ptrdiff_t;
        using reference = OrderedDict::const_reference;
        using pointer = void;

        const_iterator() = default;
        explicit const_iterator(list_citer it) : it_(it) {}
        const_iterator(iterator it) : it_(it.base()) {}

        reference operator*() const { return reference{it_->key, it_->value}; }
        const_arrow_proxy operator->() const { return const_arrow_proxy{it_->key, it_->value}; }

        const_iterator &operator++() {
            ++it_;
            return *this;
        }
        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++*this;
            return tmp;
        }
        const_iterator &operator--() {
            --it_;
            return *this;
        }
        const_iterator operator--(int) {
            const_iterator tmp = *this;
            --*this;
            return tmp;
        }

        friend bool operator==(const const_iterator &a, const const_iterator &b) {
            return a.it_ == b.it_;
        }
        friend bool operator!=(const const_iterator &a, const const_iterator &b) {
            return !(a == b);
        }

        list_citer base() const { return it_; }
    };

    // Constructors
    OrderedDict() = default;

    explicit OrderedDict(size_type bucket_count, const Hash &hash = Hash(),
                         const KeyEqual &equal = KeyEqual(), const Allocator &alloc = Allocator())
        : list_(list_alloc(alloc)), index_(bucket_count, hash, equal, map_alloc(alloc)) {}

    // Capacity
    bool empty() const noexcept { return list_.empty(); }
    size_type size() const noexcept { return index_.size(); }

    // Iteration (in insertion order)
    iterator begin() noexcept { return iterator(list_.begin()); }
    iterator end() noexcept { return iterator(list_.end()); }
    const_iterator begin() const noexcept { return const_iterator(list_.cbegin()); }
    const_iterator end() const noexcept { return const_iterator(list_.cend()); }
    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }

    // Lookup
    iterator find(const Key &k) noexcept {
        auto it = index_.find(k);
        return (it == index_.end()) ? end() : iterator(it->second);
    }
    const_iterator find(const Key &k) const noexcept {
        auto it = index_.find(k);
        return (it == index_.end()) ? end() : const_iterator(it->second);
    }
    bool contains(const Key &k) const noexcept { return index_.find(k) != index_.end(); }

    mapped_type &at(const Key &k) {
        auto it = index_.find(k);
        if (it == index_.end())
            throw std::out_of_range("OrderedDict::at");
        return it->second->value;
    }
    const mapped_type &at(const Key &k) const {
        auto it = index_.find(k);
        if (it == index_.end())
            throw std::out_of_range("OrderedDict::at");
        return it->second->value;
    }

    // Element access (like std::map/unordered_map)
    // Inserts default-constructed T if missing (at the end, preserving order).
    mapped_type &operator[](const Key &k) { return try_emplace(k).first->second; }
    mapped_type &operator[](Key &&k) { return try_emplace(std::move(k)).first->second; }

    // Modifiers
    void clear() {
        list_.clear();
        index_.clear();
    }

    // insert (copy)
    std::pair<iterator, bool> insert(const value_type &v) { return emplace(v.first, v.second); }
    // insert (move)
    std::pair<iterator, bool> insert(value_type &&v) {
        return emplace(std::move(const_cast<Key &>(v.first)), std::move(v.second));
    }

    // emplace
    template <class K, class... Args> std::pair<iterator, bool> emplace(K &&k, Args &&...args) {
        auto it = index_.find(k);
        if (it != index_.end())
            return {iterator(it->second), false};

        list_.emplace_back(Key(std::forward<K>(k)), T(std::forward<Args>(args)...));
        auto lit = std::prev(list_.end());
        index_.emplace(lit->key, lit);
        return {iterator(lit), true};
    }

    // try_emplace: does not move if key exists
    template <class K, class... Args> std::pair<iterator, bool> try_emplace(K &&k, Args &&...args) {
        auto it = index_.find(k);
        if (it != index_.end())
            return {iterator(it->second), false};

        // If no args provided, T() is used (requires default-constructible T),
        // which matches std::map/unordered_map semantics for operator[]/try_emplace.
        list_.emplace_back(Key(std::forward<K>(k)), T(std::forward<Args>(args)...));
        auto lit = std::prev(list_.end());
        index_.emplace(lit->key, lit);
        return {iterator(lit), true};
    }

    // erase by key: unchanged
    size_type erase(const Key &k) {
        auto it = index_.find(k);
        if (it == index_.end())
            return 0;
        list_.erase(it->second);
        index_.erase(it);
        return 1;
    }

    // erase by const_iterator: return const_iterator to next
    const_iterator erase(const_iterator pos) {
        auto lit = pos.base();
        index_.erase(lit->key);       // remove from index first (still valid)
        auto next = list_.erase(lit); // list::erase returns iterator to next
        return const_iterator(next);
    }

    // erase by iterator: return iterator to next
    iterator erase(iterator pos) {
        auto lit = pos.base();
        index_.erase(lit->key);       // remove from index first
        auto next = list_.erase(lit); // list::erase returns next
        return iterator(next);
    }

    // Reserve buckets in the index
    void reserve(size_type n) { index_.reserve(n); }

    // hasher/equality accessors
    hasher hash_function() const { return index_.hash_function(); }
    key_equal key_eq() const { return index_.key_eq(); }

    size_type max_size() const noexcept { return std::min(list_.max_size(), index_.max_size()); }

  private:
    list_type list_;   // list of nodes for maintaining insertion order
    index_type index_; // index of nodes for fast lookup O(1)
};

} // namespace pyslam
