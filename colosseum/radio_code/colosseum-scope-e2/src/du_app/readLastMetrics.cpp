#include "readLastMetrics.hpp"

std::vector<std::string> read_csv_file(const char *filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return lines;
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();
    return lines;
}

std::vector<std::string> read_csv_file_limit(const char *filename, unsigned int max_lines) {
    std::vector<std::string> lines;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return lines;
    }

    char ch;
    int pos;
    std::string line;
    file.seekg(-1,std::ios::end);
    pos=file.tellg();
    for(int i=0;i<pos;i++)
    {
        ch=file.get();

        if (ch == '\n'){
            if ( i > 0 ) {
                reverse(line.begin(), line.end());
                //std::cout<<line << "\n";
                lines.push_back(line);
                line.clear();
                if (lines.size() == max_lines){
                    break;
                }
            }
        }else{
            line += ch;
        }
        file.seekg(-2,std::ios::cur);
    }

    // Reverse the order of the lines to restore their original order
    std::reverse(lines.begin(), lines.end());
    //std::cout << "After reverse" << std::endl;
    //for (int i = 0; i < lines.size(); i++){
    //    std::cout << lines[i] << std::endl;
    //}
    //std::cout << " Total lines read : " << lines.size() << std::endl;
    file.close();
    return lines;
}

template<typename ValueType>
std::string stringulate(ValueType v)
{
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

// read last lines from file
std::string readLastMetricsLines(const char *file_name, int to_read, int skip_header) {

  FILE *fp;
  fp = fopen(file_name, "r");

  int max_metrics_buf = 1000;
  int min_metrics_to_send = 1;

  long unsigned int curr_ts;

  if (CSV_DEBUG) {
    curr_ts = 1680733078312 + 2000;
  }
  else {
    curr_ts = get_time_milliseconds();

  }

  if (!fp) {
    //printf("fp is NULL, filename %s\n", file_name);
    return "";
  }
  else {
    //printf("%lu: reading %s\n", curr_ts, file_name);
  }

  int tot_lines;
  for (tot_lines = 0; feof(fp) == 0; ++tot_lines) {
    fscanf(fp, "%*[^\n]\n", NULL);
  }

  // rewind file pointer
  rewind(fp);

  fclose(fp);

  // skip first lines_num - to_read lines
  // and read last to_read lines
  int tot_len = 0;
  int j = 0;
  /*
  rapidcsv::Document doc(file_name);
  const size_t rowCount = doc.GetRowCount();
  */

  //std::vector<std::string> lines = read_csv_file(file_name);
  std::vector<std::string> lines = read_csv_file_limit(file_name, to_read + 5);
  int rowCount = lines.size();
  std::string metrics_array[to_read];

  for (size_t i = rowCount - to_read; i < rowCount; ++i)
  {
    size_t metric_len = 0;
    metrics_array[j] = lines[i] + "\n";
    tot_len += metrics_array[j].length();
    j++; // increment valid metrics counter

    //std::cout << std::endl;
  }

  //output_string = (char*) calloc(tot_len + 1, sizeof(char));
  //printf("[mau] created output string, tot_len = %d, j = %d\n", tot_len, j);
  std::string output_string;


  int curr_pos = 0;
  int valid_metrics = 0;

  // copy header
  if (!skip_header) {
    output_string += metrics_array[0];
  }

  //for (int i = 0; i < to_read + 1; ++i) {
  for (int i = 0; i < j; ++i) {
    if (i == 0 && !skip_header) {
      continue;
    }

    // get metric timestamp
    long unsigned int metric_ts = 0;
    sscanf(metrics_array[i].c_str(), "%lu", &metric_ts);

    // printf("i %d, timestamp %lu, metrics_array[i] %s\n", i, metric_ts, metrics_array[i]);

    // save it if recent enough
    if (((curr_ts - metric_ts) / 1000.0) <= DELTA_TS_S) {
      // skip if empty line
      if (strcmp(metrics_array[i].c_str(), "\n") == 0 || strlen(metrics_array[i].c_str()) == 0) {
        continue;
      }

      // strip timestamp if METRICS_PRESET is 1 or more
      if (METRICS_PRESET >= 1) {

        std::string tmp_ts = stringulate(metric_ts);
	    std::string::size_type start_pos = metrics_array[i].find(tmp_ts);

        if (start_pos != std::string::npos)
           metrics_array[i].erase(start_pos, tmp_ts.length());
      }

      output_string += metrics_array[i];

      curr_pos += strlen(metrics_array[i].c_str());
      valid_metrics += 1;

      //printf("len metrics_array[i]: %d\n", strlen(metrics_array[i].c_str()));
    }
  }

  //printf("valid_metrics %d\n", valid_metrics);
  return output_string;
}

// return current time in milliseconds since the EPOCH
long unsigned int get_time_milliseconds(void) {

  long unsigned int time_ms;
  struct timespec spec;

  clock_gettime(CLOCK_REALTIME, &spec);

  time_ms  = spec.tv_sec * 1000;
  time_ms += round(spec.tv_nsec / 1.0e6);

  return time_ms;
}

std::string getCmdOption(int argc, char* argv[], const std::string& option)
{
    std::string cmd;
     for( int i = 0; i < argc; ++i)
     {
          std::string arg = argv[i];
          if(0 == arg.find(option))
          {
               std::size_t found = arg.find_first_of(option);
               cmd =arg.substr(found + 5);
               return cmd;
          }
     }
     return cmd;
}


int main(int argc, char* argv[])
{
    std::string csv_filename = getCmdOption(argc, argv, "-csv=");
    std::string ltr = getCmdOption(argc, argv, "-ltr=");
    char *metrics_string;
    int lines_to_read = atoi(ltr.c_str());
    //std::cout << csv_filename << " line_to_read " << lines_to_read << std::endl;
    std::string std_output = readLastMetricsLines(csv_filename.c_str(), lines_to_read, 1);
    std::cout << std_output << std::endl;
    return 0;
}