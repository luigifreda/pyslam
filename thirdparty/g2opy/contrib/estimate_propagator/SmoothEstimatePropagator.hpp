/**
 * This file is part of S-PTAM.
 *
 * Copyright (C) 2013-2017 Taihú Pire
 * Copyright (C) 2014-2017 Thomas Fischer
 * Copyright (C) 2016-2017 Gastón Castro
 * Copyright (C) 2017 Matias Nitsche
 * For more information see <https://github.com/lrse/sptam>
 *
 * S-PTAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * S-PTAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with S-PTAM. If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:  Taihú Pire
 *           Thomas Fischer
 *           Gastón Castro
 *           Matías Nitsche
 *
 * Laboratory of Robotics and Embedded Systems
 * Department of Computer Science
 * Faculty of Exact and Natural Sciences
 * University of Buenos Aires
 */

#ifndef __SMOOTH_ESTIMATE_PROPAGATOR_HPP__
#define __SMOOTH_ESTIMATE_PROPAGATOR_HPP__

#include <Eigen/Eigen>

// G2O
#include <g2o/config.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/estimate_propagator.h>

/*
 * SmoothEstimatePropagator propagates edge information relaxing it while moving away of fixed vertices.
 * Uses an exponential function for this, so a vertex near fixed vertices will be moved all the way accordingly
 * with edge restriction meanwhile a vertex far away will remain in the same place.
 *
 * Note that this propagator only works with VertexSE3 and EdgeSE3.
 */
namespace g2o {
class SmoothEstimatePropagator : public g2o::EstimatePropagator
{
  public:

    SmoothEstimatePropagator(g2o::SparseOptimizer* g,
                             const double& maxDistance=std::numeric_limits<double>::max(),
                             const double& maxEdgeCost=std::numeric_limits<double>::max());

    void propagate(g2o::OptimizableGraph::Vertex* v);

  private:

    struct SmoothPropagateAction : g2o::EstimatePropagator::PropagateAction {
      public:
        SmoothPropagateAction(g2o::EstimatePropagator::AdjacencyMap* adj, const double& max_distance);

        void operator()(g2o::OptimizableGraph::Edge* e_, const g2o::OptimizableGraph::VertexSet& from_, g2o::OptimizableGraph::Vertex* to_) const;

      private:
        g2o::EstimatePropagator::AdjacencyMap* adjacency;
        double maxDistance;

        Eigen::Isometry3d exponencialInterpolation(const Eigen::Isometry3d& from, const Eigen::Isometry3d& to, double step) const;
    };

    SmoothPropagateAction _smoothAction;
    g2o::EstimatePropagator::PropagateCost _treeCost;
    double _maxDistance, _maxEdgeCost;
};
}
#endif //__SMOOTH_ESTIMATE_PROPAGATOR_HPP__
