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

#include <iostream>
#include <memory>
#include <stdarg.h>
#include <type_traits>

#ifndef UNUSED_VAR
#define UNUSED_VAR(x) (void)x
#endif

namespace pyslam {

template <typename T> struct EmptyDelete {
    void operator()(T *const) const {} // do nothing
};

template <class T, class D = std::default_delete<T>>
struct SharedPtrWithDeleter : public std::shared_ptr<T> {
    explicit SharedPtrWithDeleter(T *t = nullptr) : std::shared_ptr<T>(t, D()) {}
};

// template<class Type, class TypePtr, class... Args>
// inline TypePtr NewPtr(Args&&... args) { return nullptr; }

#define DECLARE_SMART_POINTERS(Type)                                                               \
    typedef std::shared_ptr<Type> Type##Ptr;                                                       \
    typedef Type *Type##RawPtr;                                                                    \
    typedef std::weak_ptr<Type> Type##WeakPtr;                                                     \
    typedef std::shared_ptr<const Type> Type##ConstPtr;                                            \
    typedef SharedPtrWithDeleter<Type, EmptyDelete<Type>> Type##WrapPtr;                           \
    template <class... Args> inline Type##Ptr Type##NewPtr(Args &&...args) {                       \
        return std::make_shared<Type>(std::forward<Args>(args)...);                                \
    }                                                                                              \
    inline void DeletePtr(const Type##Ptr &ptr) { UNUSED_VAR(ptr); /*ptr.reset()*/ }               \
    inline Type *GetRawPtr(const Type##Ptr &ptr) { return ptr.get(); }                             \
    inline Type##WrapPtr WrapPtr(Type *ptr) { return Type##WrapPtr(ptr); }

#define DEFINE_SMART_POINTERS(Type)

#define DECLARE_POINTERS(Type)                                                                     \
    typedef Type *Type##Ptr;                                                                       \
    typedef Type *Type##RawPtr;                                                                    \
    typedef Type *Type##WeakPtr;                                                                   \
    typedef const Type *Type##ConstPtr;                                                            \
    typedef Type *Type##WrapPtr;                                                                   \
    template <class... Args> inline Type##Ptr Type##NewPtr(Args &&...args) {                       \
        return new Type(std::forward<Args>(args)...);                                              \
    }                                                                                              \
    void DeletePtr(const Type##Ptr &ptr);                                                          \
    inline Type *GetRawPtr(const Type##Ptr &ptr) { return ptr; }                                   \
    inline Type##WrapPtr WrapPtr(Type *ptr) { return ptr; }

#define DEFINE_POINTERS(Type)                                                                      \
    void DeletePtr(const Type##Ptr &ptr) {                                                         \
        delete ptr;                                                                                \
    } // The delete function needs to be put in .cc in order to provide it with full knowledge of
      // 'Type', which is only forward declared here. This allows a faster compilation by minimizing
      // the need of the full headers everywhere.

/// < N.B.: keep the following lines coherent with the content of 'Pointers.cc'

class Camera;
DECLARE_SMART_POINTERS(Camera)

class PinholeCamera;
DECLARE_SMART_POINTERS(PinholeCamera)

class CameraPose;
DECLARE_SMART_POINTERS(CameraPose)

class MapPoint;
// DECLARE_POINTERS(MapPoint)
DECLARE_SMART_POINTERS(MapPoint)

class KeyFrameGraph;
DECLARE_SMART_POINTERS(KeyFrameGraph)

class KeyFrame;
// DECLARE_POINTERS(KeyFrame)
DECLARE_SMART_POINTERS(KeyFrame)

class Frame;
// DECLARE_POINTERS(Frame)
DECLARE_SMART_POINTERS(Frame)

class Map;
// DECLARE_POINTERS(Map)
DECLARE_SMART_POINTERS(Map)

} // namespace pyslam
