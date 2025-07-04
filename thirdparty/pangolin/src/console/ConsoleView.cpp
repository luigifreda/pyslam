#include <iterator>
#include <pangolin/console/ConsoleView.h>
#include <pangolin/utils/picojson.h>
#include <pangolin/gl/gldraw.h>

namespace pangolin
{

inline Colour ParseJson(const picojson::value& val)
{
    return Colour(
        val.contains("r") ? val["r"].get<double>() : 0.0,
        val.contains("g") ? val["g"].get<double>() : 0.0,
        val.contains("b") ? val["b"].get<double>() : 0.0,
        val.contains("a") ? val["a"].get<double>() : 1.0
    );
}

inline picojson::value toJson(const Colour& colour)
{
    picojson::value ret(picojson::object_type,true);
    ret["r"] = colour.r;
    ret["g"] = colour.g;
    ret["b"] = colour.b;
    ret["a"] = colour.a;
    return ret;
}

inline std::ostream& operator<<(std::ostream& os, const Colour& colour)
{
    os << toJson(colour).serialize();
    return os;
}

inline std::istream& operator>>(std::istream& is, Colour& colour)
{
    picojson::value val;
    picojson::parse(val,is);
    colour = ParseJson(val);
    return is;
}

inline void glColour(const Colour& c)
{
    glColor4f(c.r,c.g,c.b,c.a);
}

ConsoleView::ConsoleView(ConsoleInterpreter* interpreter)
    : interpreter(interpreter),
      font(GlFont::I()),
      carat(0),
      hiding(false),
      bottom(1.0f),
      background_colour(0.2f, 0.0f, 0.0f, 0.6f),
      animation_speed(0.2)
{
    SetHandler(this);

    line_colours[ConsoleLineTypeCmd]        = Colour(1.0f,1.0f,1.0f,1.0f);
    line_colours[ConsoleLineTypeCmdOptions] = Colour(0.9f,0.9f,0.9f,1.0f);
    line_colours[ConsoleLineTypeOutput]     = Colour(0.0f,1.0f,1.0f,1.0f);
    line_colours[ConsoleLineTypeHelp]       = Colour(1.0f,0.8f,1.0f,1.0f);
    line_colours[ConsoleLineTypeStdout]     = Colour(0.0f,0.0f,1.0f,1.0f);
    line_colours[ConsoleLineTypeStderr]     = Colour(1.0f,0.8f,0.8f,1.0f);

    Var<Colour>::Attach("pangolin.console.colours.Background", background_colour);
    Var<Colour>::Attach("pangolin.console.colours.Cmd",        line_colours[ConsoleLineTypeCmd]);
    Var<Colour>::Attach("pangolin.console.colours.CmdOptions", line_colours[ConsoleLineTypeCmdOptions]);
    Var<Colour>::Attach("pangolin.console.colours.Stdout",     line_colours[ConsoleLineTypeStdout]);
    Var<Colour>::Attach("pangolin.console.colours.Stderr",     line_colours[ConsoleLineTypeStderr]);
    Var<Colour>::Attach("pangolin.console.colours.Output",     line_colours[ConsoleLineTypeOutput]);
    Var<Colour>::Attach("pangolin.console.colours.Help",       line_colours[ConsoleLineTypeHelp]);

    Var<float>::Attach("pango.console.animation_speed", animation_speed);

    AddLine("Pangolin Python Command Prompt:", ConsoleLineTypeHelp);
    AddLine("===============================", ConsoleLineTypeHelp);
}

ConsoleView::~ConsoleView() {
    delete interpreter;
}

void ConsoleView::ProcessOutputLines()
{
    // empty output queue
    ConsoleLine line_in;
    while(interpreter->PullLine(line_in))
    {
        AddLine(line_in.text, line_in.linetype);
    }
}

View& ConsoleView::ShowWithoutAnimation(bool should_show){
    Show(should_show);
    bottom = show ? 1.0 : 0.0;
    return *this;
}

View& ConsoleView::Show(bool should_show)
{
    if(should_show) {
        hiding = false;
        show = true;
    }else{
        hiding = true;
    }
    return *this;
}

void ConsoleView::ToggleShow()
{
    Show(!IsShown());
}

bool ConsoleView::IsShown() const
{
    return show && !hiding;
}

void ConsoleView::DrawLine(const ConsoleView::Line& l, int carat=-1)
{
    glColour(line_colours[l.linetype]);
    l.text.Draw();
    if(carat >= 0) {
        const double w = font.Text(l.text.str.substr(0,carat)).Width();
        glDrawLine(w,-2,w,font.Height()-4);
    }
}

void ConsoleView::Render()
{
    if(hiding) {
        bottom += (1.0f - bottom) * animation_speed;
        if(1.0 - bottom < 0.01) {
            bottom = 1.0;
            show = false;
            hiding = false;
            return;
        }
    }else{
        if(bottom > 0.01f) {
            bottom -= bottom*animation_speed;
        }else{
            bottom = 0.0f;
        }
    }

    ProcessOutputLines();

#ifndef HAVE_GLES
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_SCISSOR_BIT | GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_TRANSFORM_BIT);
#endif

    this->ActivatePixelOrthographic();
    glDisable(GL_DEPTH_TEST );
    glDisable(GL_LIGHTING);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_LINE_SMOOTH);
    glDisable( GL_COLOR_MATERIAL );
    glLineWidth(1.0);

    glEnable(GL_BLEND);
    glBlendFunc( GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA );

    glColour(background_colour);

    GLfloat verts[] = { 0.0f, (GLfloat)v.h,
                        (GLfloat)v.w, (GLfloat)v.h,
                        (GLfloat)v.w, bottom*v.h,
                        0.0f, bottom*v.h };
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, verts);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDisableClientState(GL_VERTEX_ARRAY);


    const double line_space = font.Height();
    glTranslated(10.0, 10.0 + bottom*v.h, 0.0 );
    DrawLine(current_line, carat);
    glTranslated(0.0, line_space, 0.0);
    for(size_t l=0; l < line_buffer.size(); ++l) {
        DrawLine(line_buffer[l]);
        glTranslated(0.0, line_space, 0.0);
    }

#ifndef HAVE_GLES
    glPopAttrib();
#endif
}

inline std::string CommonPrefix(const std::vector<std::string>& vec)
{
    if(!vec.size()) return "";

    size_t cmn = vec[0].size();
    for(size_t i=1; i<vec.size(); ++i) {
        cmn = std::min(vec[i].size(), cmn);
        for(size_t p=0; p < cmn; ++p) {
            if(vec[i][p] != vec[0][p]) {
                cmn = p;
                break;
            }
        }
    }

    return vec[0].substr(0,cmn);
}

void ConsoleView::Keyboard(View&, unsigned char key, int /*x*/, int /*y*/, bool pressed)
{
    static int hist_id = -1;
    static std::string prefix;
    static bool edited = true;

    if(pressed) {
        if(key=='\r') key = '\n';

        GlText& txt = current_line.text;
        const std::string cmd = txt.Text();

        if(key=='`') {
            ToggleShow();
        } else if(key=='\n') {
            interpreter->PushCommand(cmd);
            line_buffer.push_front(current_line);
            hist_id = -1;
            prefix = "";
            edited = true;
            carat = 0;
            txt.Clear();
        }else if(key=='\t') {
            std::vector<std::string> options = interpreter->Complete(cmd,100);
            if(options.size()) {
                const std::string common = CommonPrefix(options);
                if(common.size() > cmd.size()) {
                    current_line = font.Text("%s", common.c_str());
                    carat = common.size();
                }else{
                    std::stringstream s;
                    std::copy(options.begin(), options.end(), std::ostream_iterator<std::string>(s,", "));
                    AddLine(s.str(), ConsoleLineTypeCmdOptions);
                }
            }
        }else if(key==PANGO_SPECIAL + PANGO_KEY_UP) {
            if(edited) {
                prefix = cmd;
                edited = false;
            }
            Line* hist_line = GetLine(hist_id+1, ConsoleLineTypeCmd, prefix);
            if(hist_line) {
                current_line = *hist_line;
                hist_id++;
            }
        }else if(key==PANGO_SPECIAL + PANGO_KEY_DOWN) {
            if(edited) {
                prefix = cmd;
                edited = false;
            }
            Line* hist_line = GetLine(hist_id-1, ConsoleLineTypeCmd, prefix);
            if(hist_line) {
                current_line = *hist_line;
                hist_id--;
            }
        }else if(key==PANGO_SPECIAL + PANGO_KEY_LEFT) {
            if(carat > 0) carat--;
        }else if(key==PANGO_SPECIAL + PANGO_KEY_RIGHT) {
            if(carat < (int)txt.str.size()) carat++;
        }else if(key==PANGO_SPECIAL + PANGO_KEY_HOME) {
            carat = 0;
        }else if(key==PANGO_SPECIAL + PANGO_KEY_END) {
            carat = txt.Text().size();
        }else if(key=='\b') {
            if(carat > 0) {
                std::string newstr = txt.Text();
                newstr.erase(newstr.begin()+carat-1);
                txt = font.Text("%s", newstr.c_str() );
                carat--;
                edited = true;
            }
        }else if(key==127) { // delete
            if(carat < (int)txt.Text().size() ) {
                std::string newstr = txt.Text();
                newstr.erase(newstr.begin()+carat);
                txt = font.Text("%s", newstr.c_str() );
                edited = true;
            }
        }else if(key==PANGO_CTRL + 'c') {
            txt = font.Text("");
            carat = 0;
            edited = true;
        }else if(key < PANGO_SPECIAL){
            std::string newstr = txt.Text();
            newstr.insert(carat, 1, key);
            txt = font.Text("%s", newstr.c_str());
            ++carat;
            edited = true;
        }
    }
}

void ConsoleView::AddLine(const std::string& text, ConsoleLineType linetype )
{
    line_buffer.push_front( Line( font.Text("%s",text.c_str()), linetype) );
}

ConsoleView::Line* ConsoleView::GetLine(int id, ConsoleLineType line_type, const std::string& prefix )
{
    int match = 0;
    for(Line& l : line_buffer)
    {
        if(l.linetype == line_type) {
            const std::string substr = l.text.Text().substr(0,prefix.size());
            if(substr == prefix  ) {
                if(id == match) {
                    return &l;
                }else{
                    ++match;
                }
            }
        }
    }

    return nullptr;
}

}
