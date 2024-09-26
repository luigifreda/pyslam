#ifndef DBoW3_TIMERS_H
#define DBoW3_TIMERS_H


#include <chrono>
#include <string>
#include <vector>
#include <iostream>
namespace DBoW3{

//timer
struct ScopeTimer
{
    std::chrono::high_resolution_clock::time_point begin,end;

    std::string name;
    bool use;
    enum SCALE {NSEC,MSEC,SEC};
    SCALE sc;
    ScopeTimer(std::string name_,bool use_=true,SCALE _sc=MSEC)
    {
        name=name_;
        use=use_;
        sc=_sc;
        begin= std::chrono::high_resolution_clock::now();
    }
    ~ScopeTimer()
    {
        if (use){
            end= std::chrono::high_resolution_clock::now();
            double fact=1;
            std::string str;
            switch(sc)
            {
            case NSEC:fact=1;str="ns";break;
            case MSEC:fact=1e6;str="ms";break;
            case SEC:fact=1e9;str="s";break;
            };

            std::cout << "Time ("<<name<<")= "<<double(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/fact<<str<<std::endl; ;
        }
    }
};

struct ScopedTimerEvents
{
    enum SCALE {NSEC,MSEC,SEC};
    SCALE sc;
    std::vector<std::chrono::high_resolution_clock::time_point> vtimes;
    std::vector<std::string> names;
    std::string _name;

    ScopedTimerEvents(std::string name="",bool start=true,SCALE _sc=MSEC){
        if(start) add("start");sc=_sc;_name=name;
    }

    void add(std::string name){
        vtimes.push_back(std::chrono::high_resolution_clock::now());
        names.push_back(name);
    }
    void addspaces(std::vector<std::string> &str ){
        //get max size
        int m=-1;
        for(auto &s:str)m=std::max(int(s.size()),m);
        for(auto &s:str){
            while(s.size()<m) s.push_back(' ');
        }
    }

    ~ScopedTimerEvents(){
        double fact=1;
        std::string str;
        switch(sc)
        {
        case NSEC:fact=1;str="ns";break;
        case MSEC:fact=1e6;str="ms";break;
        case SEC:fact=1e9;str="s";break;
        };

        add("total");
        addspaces(names);
        for(int i=1;i<vtimes.size();i++){
            std::cout<<"Time("<<_name<<")-"<<names[i]<<" "<< double(std::chrono::duration_cast<std::chrono::nanoseconds>(vtimes[i]-vtimes[i-1]).count())/fact<<str<<" "<<double(std::chrono::duration_cast<std::chrono::nanoseconds>(vtimes[i]-vtimes[0]).count())/fact<<str<<std::endl;
        }
    }
};

struct Timer{
    enum SCALE {NSEC,MSEC,SEC};

    std::chrono::high_resolution_clock::time_point _s;
    double sum=0,n=0;
    std::string _name;
    Timer(){}

    Timer(std::string name):_name(name){}
    void setName(std::string name){_name=name;}
    void start(){_s=std::chrono::high_resolution_clock::now();}
    void end()
    {
        auto e=std::chrono::high_resolution_clock::now();
        sum+=double(std::chrono::duration_cast<std::chrono::nanoseconds>(e-_s).count());
        n++;
    }

    void print(SCALE sc=MSEC){
        double fact=1;
        std::string str;
        switch(sc)
        {
        case NSEC:fact=1;str="ns";break;
        case MSEC:fact=1e6;str="ms";break;
        case SEC:fact=1e9;str="s";break;
        };
        std::cout<<"Time("<<_name<<")= "<< ( sum/n)/fact<<str<<std::endl;
    }

};
}


#endif
