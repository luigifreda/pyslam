import os

import numpy as np
import torch
from OpenGL import GL as gl

from . import util, util_gau

_sort_buffer_xyz = None
_sort_buffer_gausid = None  # used to tell whether gaussian is reloaded


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = torch.tensor(gaus.xyz).cuda()
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


# Decide which sort to use
_sort_gaussian = None
if not torch.cuda.is_available():
    raise ImportError
_sort_gaussian = _sort_gaussian_torch


class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        raise NotImplementedError()

    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()

    def set_render_mod(self, mod: int):
        raise NotImplementedError()

    def update_camera_pose(self, camera: util.Camera):
        raise NotImplementedError()

    def update_camera_intrin(self, camera: util.Camera):
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()

    def set_render_reso(self, w, h):
        raise NotImplementedError()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.program = util.load_shaders(
            os.path.join(cur_path, "shaders/gau_vert.glsl"),
            os.path.join(cur_path, "shaders/gau_frag.glsl"),
        )

        # Vertex data for a quad
        self.quad_v = np.array([-1, 1, 1, 1, 1, -1, -1, -1], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32).reshape(2, 3)

        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.gau_bufferid = None
        self.index_bufferid = None
        
        self.fbo = None
        self.texture = None
        self.rbo = None
        self.setup_fbo(w, h)

        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(
            self.program, "gaussian_data", gaussian_data, bind_idx=0,
            buffer_id=self.gau_bufferid
        )
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(self.program, "gi", index, bind_idx=1,
                                                           buffer_id=self.index_bufferid)
        return

    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")
                                
    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)     
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES,
            len(self.quad_f.reshape(-1)),
            gl.GL_UNSIGNED_INT,
            None,
            num_gau,
        )
        
    def setup_fbo(self, width, height):
        print('OpenGLRenderer: Initializing FBO')            
        # clear the previous FBO
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo])
        if self.texture is not None:
            gl.glDeleteTextures(1, [self.texture])
        if self.rbo is not None:
            gl.glDeleteRenderbuffers(1, [self.rbo])
        
        self.width, self.height = width, height
        # Create and bind the FBO
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        # Create a texture to render to
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # Attach the texture to the FBO
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture, 0)

        # Create a renderbuffer object for depth and stencil attachment (optional)
        self.rbo = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self.rbo)

        # Check if the FBO is complete
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("OpenGLRenderer: Framebuffer is not complete")
        
        # Unbind the FBO
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        

    def draw_fbo(self, width, height):
        if self.width != width or self.height != height:
            self.setup_fbo(width, height)
        
        # Bind the FBO for off-screen rendering
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.width, self.height)
        
        # Clear the FBO
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Use the shader program and bind the VAO
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        
        # Draw the elements
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES,
            len(self.quad_f.reshape(-1)),
            gl.GL_UNSIGNED_INT,
            None,
            num_gau,
        )
        # # Unbind the FBO to render to the default framebuffer
        # gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        
    def check_gl_error(self, operation="operation"):
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGLRenderer: OpenGL error after {operation}: {error}")        
            
    def check_framebuffer_status(self):
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        #print(f'Framebuffer status: {status}')
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"OpenGLRenderer: Framebuffer incomplete: {status}") 
            if status == gl.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                print("OpenGLRenderer: Framebuffer incomplete attachment")
            elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                print("OpenGLRenderer: Framebuffer missing attachment")
            elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                print("OpenGLRenderer: Framebuffer incomplete draw buffer")
            elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                print("OpenGLRenderer: Framebuffer incomplete read buffer")
            elif status == gl.GL_FRAMEBUFFER_UNSUPPORTED:
                print("OpenGLRenderer: Framebuffer unsupported")
            elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                print("OpenGLRenderer: Framebuffer incomplete multisample")
            elif status == gl.GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
                print("OpenGLRenderer: Framebuffer incomplete layer targets")
            return
