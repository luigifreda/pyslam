/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <pangolin/var/varextra.h>
#include <pangolin/var/varstate.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/picojson.h>
#include <pangolin/utils/transform.h>

#include <iostream>
#include <fstream>

using namespace std;

namespace pangolin
{

VarState& VarState::I() {
    static VarState singleton;
    return singleton;
}

VarState::VarState()
    : varHasChanged(false)
{
}

VarState::~VarState() {
    Clear();
}

void VarState::Clear() {
    for(VarStoreContainer::iterator i = vars.begin(); i != vars.end(); ++i) {
        delete i->second;
    }
    vars.clear();
    var_adds.clear();
}

void ProcessHistoricCallbacks(NewVarCallbackFn callback, void* data, const std::string& filter)
{
    for (VarState::VarStoreAdditions::const_iterator i = VarState::I().var_adds.begin(); i != VarState::I().var_adds.end(); ++i)
    {
        const std::string& name = *i;
        if (StartsWith(name, filter)) {
            callback(data, name, *VarState::I()[name], false);
        }
    }
}

void RegisterNewVarCallback(NewVarCallbackFn callback, void* data, const std::string& filter)
{
    VarState::I().new_var_callbacks.push_back(NewVarCallback(filter,callback,data));
}

void RegisterGuiVarChangedCallback(GuiVarChangedCallbackFn callback, void* data, const std::string& filter)
{
    VarState::I().gui_var_changed_callbacks.push_back(GuiVarChangedCallback(filter,callback,data));
}

// Recursively expand val
string ProcessVal(const string& val )
{
    return Transform(val, [](const std::string& k) -> std::string {
        if( VarState::I().Exists(k) ) {
             return VarState::I()[k]->str->Get();
        }else{
            return std::string("#");
        }
    });
}

void AddVar(const std::string& name, const string& val )
{
    const std::string full = ProcessVal(val);

    VarValueGeneric*& v = VarState::I()[name];
    if(!v) {
        VarValue<std::string>* nv = new VarValue<std::string>(val);
        InitialiseNewVarMetaGeneric<std::string>(*nv, name);
        v = nv;
    }
    v->str->Set(full);
}

#ifdef ALIAS
void AddAlias(const string& alias, const string& name)
{
    std::map<std::string,_Var*>::iterator vi = vars.find(name);
    
    if( vi != vars.end() )
    {
        _Var * v = vi->second;
        vars[alias].create(v->val,v->val_default,v->type_name);
        vars[alias].Meta().friendly = alias;
        v->generic = false;
        vars[alias].generic = false;
    }else{
        pango_print_error("Variable %s does not exist to alias.\n", name);
    }
}
#endif

void ParseVarsFile(const string& filename)
{
    ifstream f(filename.c_str());
    
    if( f.is_open() )
    {
        while( !f.bad() && !f.eof())
        {
            const int c = f.peek();
            
            if( isspace(c) )
            {
                // ignore leading whitespace
                f.get();
            }else{
                if( c == '#' || c == '%' )
                {
                    // ignore lines starting # or %
                    string comment;
                    getline(f,comment);
                }else{
                    // Otherwise, find name and value, seperated by '=' and ';'
                    string name;
                    string val;
                    getline(f,name,'=');
                    getline(f,val,';');
                    name = Trim(name, " \t\n\r");
                    val = Trim(val, " \t\n\r");
                    
                    if( name.size() >0 && val.size() > 0 )
                    {
                        if( !val.substr(0,1).compare("@") )
                        {
#ifdef ALIAS
                            AddAlias(name,val.substr(1));
#endif
                        }else{
                            AddVar(name,val);
                        }
                    }
                }
            }
        }
        f.close();
    }else{
        cerr << "Unable to open '" << filename << "' for configuration data" << endl;
    }
}

PANGOLIN_EXPORT
void LoadJsonFile(const std::string& filename, const string &prefix)
{
    bool some_change = false;

    picojson::value file_json(picojson::object_type,true);
    std::ifstream f(filename);
    if(f.is_open()) {
        const std::string err = picojson::parse(file_json,f);
        if(err.empty()) {
            if(file_json.contains("vars") ) {
                picojson::value vars = file_json["vars"];
                if(vars.is<picojson::object>()) {
                    for(picojson::object::iterator
                        i = vars.get<picojson::object>().begin();
                        i!= vars.get<picojson::object>().end();
                        ++i)
                    {
                        const std::string& name = i->first;
                        if(pangolin::StartsWith(name, prefix)) {
                            const std::string& val = i->second.get<std::string>();

                            VarValueGeneric*& v = VarState::I()[name];
                            if(!v) {
                                VarValue<std::string>* nv = new VarValue<std::string>(val);
                                InitialiseNewVarMetaGeneric<std::string>(*nv, name);
                                v = nv;
                            }else{
                                v->str->Set(val);
                                v->Meta().gui_changed = true;
                            }
                            some_change = true;
                        }
                    }
                }
            }
        }else{
            pango_print_error("%s\n", err.c_str());
        }
    }else{
        pango_print_error("Unable to load vars from %s\n", filename.c_str());
    }

    if(some_change) {
        FlagVarChanged();
    }
}

PANGOLIN_EXPORT
void SaveJsonFile(const std::string& filename, const string &prefix)
{
    picojson::value vars(picojson::object_type,true);

    for(VarState::VarStoreAdditions::const_iterator
        i  = VarState::I().var_adds.begin();
        i != VarState::I().var_adds.end();
        ++i)
    {
        const std::string& name = *i;
        if(StartsWith(name,prefix)) {
            try{
                const std::string val = VarState::I()[name]->str->Get();
                vars[name] = val;
            }catch(const BadInputException&)
            {
                // Ignore things we can't serialise
            }
        }
    }

    picojson::value file_json(picojson::object_type,true);
    file_json["pangolin_version"] = PANGOLIN_VERSION_STRING;
    file_json["vars"] = vars;

    std::ofstream f(filename);
    if(f.is_open()) {
        f << file_json.serialize(true);
    }else{
        pango_print_error("Unable to serialise to %s\n", filename.c_str());
    }
}

}
