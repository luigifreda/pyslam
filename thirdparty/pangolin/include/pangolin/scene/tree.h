#pragma once

#include <map>
#include <pangolin/display/opengl_render_state.h>

namespace pangolin {

template<typename T, typename TEdge>
struct TreeNode
{
    struct Edge
    {
        TEdge parent_child;
        TreeNode node;
    };

    T item;
    std::vector<Edge> edges;
};

template<typename T, typename TEdge>
using NodeFunction = std::function<void(TreeNode<T,TEdge>&,const TEdge&)>;

//template<typename T, typename TEdge>
//void VisitDepthFirst(TreeNode<T,TEdge>& node, const NodeFunction<T,TEdge>& func, const TEdge& T_root_node = TEdge())
//{
//    func(node, T_root_node);
//    for(auto& e : node.edges) {
//        const TEdge T_root_child = T_root_node * e.parent_child;
//        VisitDepthFirst(e.node, func, T_root_child);
//    }
//}

//void Eg()
//{
//    using RenderNode = TreeNode<std::shared_ptr<Renderable>,OpenGlMatrix>;

//    RenderNode root;
//    VisitDepthFirst<std::shared_ptr<Renderable>,OpenGlMatrix>(
//        root, [](RenderNode& node, const OpenGlMatrix& T_root_node) {
//        if(node.item) {
//            node.item->DoRender();
//        }
//    }, IdentityMatrix());

//}


}
