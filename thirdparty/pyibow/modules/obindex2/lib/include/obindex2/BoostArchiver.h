/*
 * This file is part of PLVS
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */


#ifndef BOOST_ARCHIVER_H
#define BOOST_ARCHIVER_H

#ifndef UNUSED_VAR
#define UNUSED_VAR(x) (void)x
#endif 

#include <boost/serialization/access.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/list.hpp>

#include <boost/serialization/deque.hpp>
#include <boost/serialization/stack.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/unordered_map.hpp>

#include <boost/serialization/queue.hpp> // NOTE: this must stay after deque.hpp inclusion!

#include <boost/serialization/shared_ptr.hpp>

#include <boost/serialization/utility.hpp>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/base_object.hpp>

#include <opencv2/core/core.hpp>

#include <tuple>

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost
{
    namespace serialization 
    {

        template<class Archive>
        void serialize(Archive& ar, unsigned char& c, const unsigned int version) {
            unsigned int temp = c; // Promote to unsigned int for serialization
            ar & temp;
            c = static_cast<unsigned char>(temp);
        }

        /* serialization for CV Point_ */
        template<class Archive, typename Tp>
        void serialize(Archive &ar, cv::Point_<Tp> &p, const unsigned int file_version)
        {
            UNUSED_VAR(file_version);

            ar & p.x;
            ar & p.y;
        }   

        /* serialization for CV Point3_ */
        template<class Archive, typename Tp>
        void serialize(Archive &ar, cv::Point3_<Tp> &p, const unsigned int file_version)
        {
            UNUSED_VAR(file_version);

            ar & p.x;
            ar & p.y;
            ar & p.z;
        }    

        /* serialization for CV KeyPoint */
        template<class Archive>
        void serialize(Archive &ar, cv::KeyPoint &kp, const unsigned int file_version)
        {
            UNUSED_VAR(file_version);

            ar & kp.pt.x;
            ar & kp.pt.y;
            ar & kp.size;
            ar & kp.angle;
            ar & kp.response;
            ar & kp.octave;
            ar & kp.class_id;
        }    

        /* serialization for CV Mat */
        template<class Archive>
        void save(Archive &ar, const cv::Mat &m, const unsigned int file_version)
        {
            UNUSED_VAR(file_version);

            cv::Mat m_ = m;
            if (!m.isContinuous())
                m_ = m.clone();
            size_t elem_size = m_.elemSize();
            size_t elem_type = m_.type();
            ar & m_.cols;
            ar & m_.rows;
            ar & elem_size;
            ar & elem_type;

            const size_t data_size = m_.cols * m_.rows * elem_size;

            ar & boost::serialization::make_array(m_.ptr(), data_size);
        }
        template<class Archive>
        void load(Archive & ar, cv::Mat& m, const unsigned int version)
        {
            UNUSED_VAR(version);

            int cols, rows;
            size_t elem_size, elem_type;

            ar & cols;
            ar & rows;
            ar & elem_size;
            ar & elem_type;

            m.create(rows, cols, elem_type);
            size_t data_size = m.cols * m.rows * elem_size;

            ar & boost::serialization::make_array(m.ptr(), data_size);
        } 


        template<class Archive>
        void serializeMatrix(Archive &ar, cv::Mat& mat, const unsigned int version)
        {
            UNUSED_VAR(version);

            int cols, rows, type;
            bool continuous;

            if (Archive::is_saving::value) {
                cols = mat.cols; rows = mat.rows; type = mat.type();
                continuous = mat.isContinuous();
            }

            ar & cols & rows & type & continuous;
            if (Archive::is_loading::value)
                mat.create(rows, cols, type);

            if (continuous) {
                const unsigned int data_size = rows * cols * mat.elemSize();
                ar & boost::serialization::make_array(mat.ptr(), data_size);
            } else {
                const unsigned int row_size = cols*mat.elemSize();
                for (int i = 0; i < rows; i++) {
                    ar & boost::serialization::make_array(mat.ptr(i), row_size);
                }
            }
        }

        template<class Archive>
        void serializeMatrix(Archive& ar, const cv::Mat& mat, const unsigned int version)
        {
            UNUSED_VAR(version);

            cv::Mat matAux = mat;

            serializeMatrix(ar, matAux,version);

            if (Archive::is_loading::value)
            {
                cv::Mat* ptr;
                ptr = (cv::Mat*)( &mat );
                *ptr = matAux;
            }
        }  
        
        // from https://github.com/Sydius/serialize-tuple/blob/master/serialize_tuple.h
        // serialize tuple 
        template<unsigned int N>
        struct Serialize
        {
            template<class Archive, typename... Args>
            static void serialize(Archive & ar, std::tuple<Args...> & t, const unsigned int version)
            {
                ar & std::get<N-1>(t);
                Serialize<N-1>::serialize(ar, t, version);
            }
        };

        template<>
        struct Serialize<0>
        {
            template<class Archive, typename... Args>
            static void serialize(Archive & ar, std::tuple<Args...> & t, const unsigned int version)
            {
                (void) ar;
                (void) t;
                (void) version;
            }
        };

        template<class Archive, typename... Args>
        void serialize(Archive & ar, std::tuple<Args...> & t, const unsigned int version)
        {
            Serialize<sizeof...(Args)>::serialize(ar, t, version);
        }

    }
}

#endif // BOOST_ARCHIVER_H
