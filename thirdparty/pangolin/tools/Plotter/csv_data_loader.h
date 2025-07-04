#pragma once

#include <pangolin/platform.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

class CsvDataLoader
{
public:
    CsvDataLoader(const std::vector<std::string>& csv_files, char delim = ',')
        : delim(delim)
    {
        for(const auto& f : csv_files) {
            if(f == "-") {
                streams.push_back(&std::cin);
            }else{
                std::ifstream* pFile = new std::ifstream(f);
                owned_streams.emplace_back(pFile);
                streams.push_back(pFile);
                PANGO_ASSERT(pFile->is_open());
            }
        }
    }

    bool SkipStreamRows(const std::vector<size_t>& rows_to_skip)
    {
        if(rows_to_skip.size()) {
            PANGO_ASSERT(rows_to_skip.size() == streams.size());
            std::vector<std::string> dummy_row;

            for(size_t i=0; i < streams.size(); ++i) {
                for(size_t r=0; r < rows_to_skip[i]; ++r) {
                    if(!AppendColumns(dummy_row, *streams[i], delim)) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    bool ReadRow(std::vector<std::string>& row)
    {
        row.clear();

        for(auto& s : streams) {
            if(!AppendColumns(row, *s, delim)) {
                return false;
            }
        }

        return true;
    }

private:
    static bool AppendColumns(std::vector<std::string>& cols, std::istream& s, char delim)
    {
        // Read line from stream
        std::string row;
        std::getline(s,row);

        // Failure if no lines to read
        if(!s.good()) return false;

        std::stringstream row_stream(row);
        std::string cell;

        // Read cells
        while(std::getline(row_stream, cell, delim)) {
            cols.push_back(cell);
        }

        // Check for an empty trailing cell
        if (!row_stream && cell.empty()) {
            cols.push_back("");
        }

        return true;
    }

    char delim;
    std::vector<std::istream*> streams;
    std::vector<std::unique_ptr<std::istream>> owned_streams;
};
