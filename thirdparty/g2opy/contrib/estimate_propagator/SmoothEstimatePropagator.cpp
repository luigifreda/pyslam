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

#include "SmoothEstimatePropagator.hpp"

#include <Eigen/Eigen>

// OpenCV
//#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>

using namespace std;
//using namespace cv;
namespace g2o {
SmoothEstimatePropagator::SmoothEstimatePropagator(g2o::SparseOptimizer* g,
                         const double& maxDistance,
                         const double& maxEdgeCost)
  : g2o::EstimatePropagator(g), _smoothAction(&_adjacencyMap,maxDistance), _treeCost(g),
    _maxDistance(maxDistance), _maxEdgeCost(maxEdgeCost) {}

void SmoothEstimatePropagator::propagate(g2o::OptimizableGraph::Vertex* v)
{ g2o::EstimatePropagator::propagate(v, _treeCost, _smoothAction, _maxDistance, _maxEdgeCost); }

SmoothEstimatePropagator::SmoothPropagateAction::
  SmoothPropagateAction(g2o::EstimatePropagator::AdjacencyMap* adj, const double& max_distance)
  : adjacency(adj), maxDistance(max_distance){}

void SmoothEstimatePropagator::SmoothPropagateAction::
  operator()(g2o::OptimizableGraph::Edge* e_, const g2o::OptimizableGraph::VertexSet& from_, g2o::OptimizableGraph::Vertex* to_) const
{
  if (to_->fixed())
    return;

  // Static cast to SE3, this must be ensure beforehand using the propagator.
  g2o::VertexSE3* from = static_cast<g2o::VertexSE3*>(e_->vertex(0));
  g2o::VertexSE3* to = static_cast<g2o::VertexSE3*>(e_->vertex(1));
  g2o::EdgeSE3* e = static_cast<g2o::EdgeSE3*>(e_);

  if (from_.count(from) > 0){
    auto entry = adjacency->find(to);
    double distance = entry == adjacency->end() ? 0 : entry->second.distance(); // this shouldnt happen! "to" must be on the adjacency map
    to->setEstimate(exponencialInterpolation(to->estimate(), from->estimate() * e->measurement(), distance));
  }else{
    auto entry = adjacency->find(from);
    double distance = entry == adjacency->end() ? 0 : entry->second.distance(); // this shouldnt happen! "to" must be on the adjacency map
    from->setEstimate(exponencialInterpolation(from->estimate(), to->estimate() * e->measurement().inverse(), distance));
  }
}

Eigen::Isometry3d SmoothEstimatePropagator::SmoothPropagateAction::
  exponencialInterpolation(const Eigen::Isometry3d& from, const Eigen::Isometry3d& to, double step) const
{
  Eigen::Isometry3d res;

  double maxdist = maxDistance-2;

  // step goes from 1 to maxDistance, we need x from 0 to 1 in a linear way.
  double x = 1 - ((maxdist - (step-1))/maxdist);
  // alpha in [0, inf) describes the explonential ramp "steepness"
  double alpha = 50;
  // exponential ramp from 0 to 1
  double exp_ramp = 1 - (x/(1+alpha*(1-x)));

  // using quaternion representation and slerp for interpolate between from and to isometry transformations
  res.linear() = (Eigen::Quaterniond(from.rotation()).slerp(exp_ramp, Eigen::Quaterniond(to.rotation()))).toRotationMatrix();
  res.translation() = (1 - exp_ramp) * from.translation() + exp_ramp * to.translation();

  return res;
}
}