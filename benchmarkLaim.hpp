#ifndef BENCHMARK_LAIM_HPP
#define BENCHMARK_LAIM_HPP

#include <string>
#include <vector>
#include <fstream>
#include <chrono>

#include <iostream>

namespace Bench {

    template<typename Func, typename... Args>
    int64_t getCodeExecutionTime(Func func, Args... args) {
        auto sTime = std::chrono::high_resolution_clock::now();
        func(args...);
        auto eTime = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>( eTime - sTime ).count();
    }

    template<typename Param, typename Func>
    class BenchmarkResult {

        private:

            std::string funcName;
            std::vector<Param> paramV;
            std::vector<int64_t> avgTimeV;

        public:

            Func func; // direct call

            BenchmarkResult(Func func_, std::string funcName_): funcName(funcName_), func(func_) { };

            void addResult(Param param, int64_t avgTime) {
                paramV.push_back(param);
                avgTimeV.push_back(avgTime);
            }

            std::string toJS() {
                std::string x, y;
                for (Param d: paramV)        { x += std::to_string(d); x += ", "; }
                for (int64_t d: avgTimeV)    { y += std::to_string(d); y += ", "; }
                return "{ x: [" + x + "], y: [" + y + "], " + "name: \"" + funcName + "\", }, ";
            }
    };

    template<typename Param, typename Func>
    class Benchmark {

        private:

            std::string benchmarkName;
            std::string parameterName;
            std::vector<BenchmarkResult<Param, Func>> results;
            bool isBlocked = false;
            unsigned int numberOfRepeat;
            int64_t maxEcecutionTimeOfTheFullIteration;

        public:

            Benchmark(std::string benchName, std::string paramName, unsigned int repeat, int64_t iterTimeLimit): 
                benchmarkName(benchName),
                parameterName(paramName), 
                numberOfRepeat(repeat),
                maxEcecutionTimeOfTheFullIteration(iterTimeLimit)
                { };

            void addFunction(Func func, std::string funcName) { 
                results.emplace_back(BenchmarkResult<Param, Func>(func, funcName));
            }

            template<typename... Args>
            void addFunction(Func func, std::string funcName, Args... args) {
                addFunction(func, funcName);
                addFunction(args...);
            }

            template<class Handler, typename... Args>
            void runWithParam(Param param, Args... args) {
                if (isBlocked) return;
                Handler handler;

                for (BenchmarkResult<Param, Func>& benchmarkResult: results) {
                    int64_t avgTime = 0;
                    for (unsigned int i = 0; i < numberOfRepeat; ++i) {
                        handler.setup(args...);
                        avgTime += getCodeExecutionTime(benchmarkResult.func, args...);
                    }
                    benchmarkResult.addResult(param, static_cast<int64_t> (avgTime / numberOfRepeat));
                    if (avgTime >= maxEcecutionTimeOfTheFullIteration) { 
                        setBlockingStatus(true);
                    }
                }
            }

            void setBlockingStatus(bool value) { isBlocked = value; }
            void setNumberOfRepeat(unsigned int value) { numberOfRepeat = value; }

            std::string toJS() {
                std::string res;

                for (BenchmarkResult<Param, Func>& benchmarkResult: results) {
                    res += benchmarkResult.toJS();
                }

                return  "{ benchmarkName: \"" + benchmarkName + 
                        "\", parameterName: \"" + parameterName + 
                        "\", data: [ " + res + "]}, ";
            }
            
    };

    class BenchmarkReport {

        private: 
            std::ofstream jsReportFile;

        public:

            BenchmarkReport(std::string filePath) {
                jsReportFile.open(filePath);
                jsReportFile << "const benchmark = [ ";
            }   

            ~BenchmarkReport() {
                jsReportFile << "];\n";
                jsReportFile.close();
            }

            void write(std::string s) {
                jsReportFile << s;
            }

            template<typename... Args>
            void write(std::string s, Args... args) {
                write(s);
                write(args...);
            }
    };

    class EmptyHandler {

        public:

            template<typename... Args>
            void setup(Args... args) { };
    };
}

#endif
