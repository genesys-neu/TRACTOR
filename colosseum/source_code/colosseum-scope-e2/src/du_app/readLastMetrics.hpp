#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "csv_reader.h"

#define CSV_DEBUG 0

std::string readLastMetricsLines(const char *file_name, int to_read, int skip_header);
