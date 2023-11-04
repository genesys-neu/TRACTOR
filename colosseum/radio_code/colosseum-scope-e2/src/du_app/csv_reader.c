#include <dirent.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csv_reader.h"


// read line into structure
void readMetrics(FILE *fp, bs_metrics_t *metrics) {

  fscanf(fp, "%lu,%d,%llu,%" PRIu16 ",,\
    %" PRIu8 ",%" PRIu8 ",%" PRIu8 ",%f,%" PRIu8 ",,\
    %f,%" PRIu8 ",%" PRIu32 ",%lf,%" PRIu16 ",%f,%f,,\
    %f,%" PRIu8 ",%" PRIu32 ",%lf,%" PRIu16 ",%f,%f,%f,%" PRIu8 ",,\
    %" PRIu16 ",%" PRIu16 ",,\
    %" PRIu8 ",%" PRIu8 ",%" PRIu8 ",%f",
    &metrics->timestamp, &metrics->num_ues, &metrics->imsi, &metrics->rnti,
    &metrics->slicing_enabled, &metrics->slice_id, &metrics->slice_prb, &metrics->power_multiplier, &metrics->scheduling_policy,
    &metrics->dl_mcs, &metrics->dl_n_samples, &metrics->dl_buffer_bytes, &metrics->tx_brate_downlink_Mbps, &metrics->tx_pkts_downlink, &metrics->tx_errors_downlink_perc, &metrics->dl_cqi,
    &metrics->ul_mcs, &metrics->ul_n_samples, &metrics->ul_buffer_bytes, &metrics->rx_brate_downlink_Mbps, &metrics->rx_pkts_downlink, &metrics->rx_errors_downlink_perc, &metrics->ul_rssi, &metrics->ul_sinr, &metrics->phr,
    &metrics->sum_requested_prbs, &metrics->sum_granted_prbs,
    &metrics->dl_pmi, &metrics->dl_ri, &metrics->ul_n, &metrics->ul_turbo_iters);
}


// read metrics into bs_metrics structure and assemble line to send
// done for future extension in which metrics are selectively sent
// metrics_preset selects which metrics to send. E.g., 0: send all metrics
void readMetricsInteractive(FILE *fp, char (*output_string)[MAX_BUF_SIZE], int metrics_preset) {

  bs_metrics_t metrics;
  readMetrics(fp, &metrics);

  float ratio_granted_req_prb;
  char selected_metrics[MAX_BUF_SIZE];

  // we want to send all the time for preset 0 or 2
  if (metrics.sum_requested_prbs > 0 || (metrics.sum_requested_prbs == 0 && metrics.sum_granted_prbs > 0) || CSV_DEBUG || metrics_preset == 2 || metrics_preset == 0) {
    switch(metrics_preset) {
      case 0:
        sprintf(selected_metrics, "%lu,%d,%llu,%" PRIu16 ","\
          "%" PRIu8 ",%" PRIu8 ",%" PRIu8 ",%.2f,%" PRIu8 ","\
          "%.2f,%" PRIu8 ",%" PRIu32 ",%.2lf,%" PRIu16 ",%.2f,%.2f,"\
          "%.2f,%" PRIu8 ",%" PRIu32 ",%.2lf,%" PRIu16 ",%.2f,%.2f,%.2f,%" PRIu8 ","\
          "%" PRIu16 ",%" PRIu16 ","\
          "%" PRIu8 ",%" PRIu8 ",%" PRIu8 ",%.2f",
          metrics.timestamp, metrics.num_ues, metrics.imsi, metrics.rnti,
          metrics.slicing_enabled, metrics.slice_id, metrics.slice_prb, metrics.power_multiplier, metrics.scheduling_policy,
          metrics.dl_mcs, metrics.dl_n_samples, metrics.dl_buffer_bytes, metrics.tx_brate_downlink_Mbps, metrics.tx_pkts_downlink, metrics.tx_errors_downlink_perc, metrics.dl_cqi,
          metrics.ul_mcs, metrics.ul_n_samples, metrics.ul_buffer_bytes, metrics.rx_brate_downlink_Mbps, metrics.rx_pkts_downlink, metrics.rx_errors_downlink_perc, metrics.ul_rssi, metrics.ul_sinr, metrics.phr,
          metrics.sum_requested_prbs, metrics.sum_granted_prbs,
          metrics.dl_pmi, metrics.dl_ri, metrics.ul_n, metrics.ul_turbo_iters);

          break;
      case 1:
        if (metrics.sum_requested_prbs > 0) {
          ratio_granted_req_prb = ((float) metrics.sum_granted_prbs) / ((float) metrics.sum_requested_prbs);
        }
        else {
          ratio_granted_req_prb = 1;
        }

        if (ratio_granted_req_prb > 1) {
          ratio_granted_req_prb = 1;
        }
        else if (CSV_DEBUG && isnan(ratio_granted_req_prb)) {
          ratio_granted_req_prb = 0;
        }

        //////////////////////////////////////////////////////////////////////////
        //
        // ordering metrics in same order as agent parser (numbers mark the order)
        // timestamp is only used for metrics freshness and removed before sending data
        //
        // metric_dict = {"dl_buffer [bytes]": 1, "tx_brate downlink [Mbps]": 2,
        //                "ratio_req_granted": 3, "slice_id": 0, "slice_prb": 4}
        //
        //////////////////////////////////////////////////////////////////////////

        sprintf(selected_metrics, "%lu,%" PRIu8 ",%" PRIu32 ",%.2lf,%.2f,%" PRIu8 ",%" PRIu16 "",
          metrics.timestamp, metrics.slice_id, metrics.dl_buffer_bytes, metrics.tx_brate_downlink_Mbps,
          ratio_granted_req_prb, metrics.slice_prb, metrics.tx_pkts_downlink);

        printf("selected_metrics %s\n", selected_metrics);

        break;

      //Initial case for capturing useful metrics for sensing related xApp development
      //This is currently not working. Somehow it is not reading the correct entries.
      //This adds overhead, but for now we will just send all metrics (case 0)
      //We will take care of parsing on the recieve side
      case 2:
        sprintf(selected_metrics, "%lu,"\
          "%.2f,%" PRIu8 ",%" PRIu32 ",%.2lf,%" PRIu16 ",%.2f,%.2f,"\
          "%.2f,%" PRIu8 ",%" PRIu32 ",%.2lf,%" PRIu16 ",%.2f,%.2f,%" PRIu8 ","\
          "%" PRIu16 ",%" PRIu16 ","\
          "%.2f",
          metrics.timestamp,
          metrics.dl_mcs, metrics.dl_n_samples, metrics.dl_buffer_bytes, metrics.tx_brate_downlink_Mbps, metrics.tx_pkts_downlink, metrics.tx_errors_downlink_perc, metrics.dl_cqi,
          metrics.ul_mcs, metrics.ul_n_samples, metrics.ul_buffer_bytes, metrics.rx_brate_downlink_Mbps, metrics.rx_pkts_downlink, metrics.rx_errors_downlink_perc, metrics.ul_sinr, metrics.phr,
          metrics.sum_requested_prbs, metrics.sum_granted_prbs,
          metrics.ul_turbo_iters);

        break;

      default:
        printf("readMetricsInteractive: Preset %d unknown\n", metrics_preset);
    }
  }

  strcpy(*output_string, selected_metrics);
}

void append_uint_to_str(unsigned int num, char* str) {
    char *temp = calloc(32, sizeof(char));
    sprintf(temp, "%u", num);
    strcat(str, temp);
    free(temp);
}

void append_string(char **dest, char *src) {
    size_t dest_len = strlen(*dest);
    size_t src_len = strlen(src);
    if (dest_len == 0) {
        *dest = calloc(src_len + 1, sizeof(char));
    }else{
        *dest = realloc(*dest, (dest_len + src_len + 1) * sizeof(char)); // +1 for null-terminator
    }
    strcat(*dest, src); //note: strcat adds the null character at the end of the concatenated string
}

#define BUFSIZE 512
// read last lines from file
void readLastMetricsLines(char *file_name, int to_read, char **output_string, int skip_header) {

  int max_metrics_buf = MAX_BUF_SIZE;
  int min_metrics_to_send = 1;

  long unsigned int curr_ts;

  if (CSV_DEBUG) {
    curr_ts = 1602173692928;
  }
  else {
    curr_ts = get_time_milliseconds();
  }

  char *cmd_prefix = "/root/radio_code/colosseum-scope-e2/src/du_app/readLastMetrics.o -csv=";
  char *cmd = (char*) malloc((strlen(cmd_prefix)+1)*sizeof(char));
  strcpy(cmd, cmd_prefix);

  append_string(&cmd, file_name);
  append_string(&cmd, " -ltr=");
  append_uint_to_str((unsigned int) to_read, cmd);
  printf("[mau] Running cmd %s\n", cmd);


  size_t valid_metrics = 0;
  char buf[BUFSIZE];
  FILE *p_fp;

  if ((p_fp = popen(cmd, "r")) == NULL) {
      printf("Error opening pipe!\n");
      return -1;
  }

  while (fgets(buf, BUFSIZE, p_fp) != NULL) {
      if (!(strcmp(buf, "\n") == 0)){
          append_string(output_string, buf);
          valid_metrics++;
          printf("--> %s", buf);
      }

  }

  if (strlen(*output_string) > 1){
    size_t num_chars = strlen(*output_string);
    (*output_string)[num_chars] = '\0';
  }

  if (pclose(p_fp)) {
      printf("Command not found or exited with error status\n");
      return -1;
  }

  printf("[mau] valid_metrics %d\nTot. Return output: %s \n", valid_metrics, *output_string);


  /*
  if (valid_metrics < 1) {
    printf("Freeing inside readLastMetricsLines\n");
    if (strlen(*output_string) > 1){
        free(*output_string);
        *output_string = NULL;
        printf("Freed\n");
    }
  }
    //the following we don't need it (this is just to resize the buffer if less lines have been read, but in our case is allocated dynamically)

  else if (valid_metrics < to_read) {
    printf("Reallocating inside readLastMetricsLines\n");
    // reallocate output_string accordingly
    *output_string = (char*) realloc(*output_string, (strlen(*output_string) + 1) * sizeof(char*));
    printf("[mau] output_string %d, after realloc %d\n", strlen(*output_string), (strlen(*output_string) + 1));
    printf("Reallocated\n");
  }
  */
}


// get content of specified directory

int getDirContent(char *directory_name, char (*dir_content)[MAX_BUF_SIZE]) {

  DIR *ptr;
  struct dirent *directory;

  ptr = opendir(directory_name);

  int num_el = 0;
  while((directory = readdir(ptr)) != NULL) {
    char tmp_str[MAX_BUF_SIZE];
    sscanf(directory->d_name, "%s", tmp_str);

    char* token = strtok(tmp_str, "_");

    char imsi[MAX_BUF_SIZE];
    strcpy(imsi, token);
    token = strtok(NULL, "_");
    if (token != NULL) {
      if (strlen(imsi) > 4 && strcmp(token, "metrics.csv") == 0) {
        strcpy(dir_content[num_el++], directory->d_name);
      }
    }
  }

  closedir(ptr);

  return num_el;
}

/*
// get content of specified directory
int getDirContent(char *directory_name, char (*dir_content)[MAX_BUF_SIZE]) {

  DIR *ptr;
  struct dirent *directory;

  // open the specified directory
  ptr = opendir(directory_name);

  int num_el = 0;
  // iterate over each file in the directory
  while((directory = readdir(ptr)) != NULL) {
    // extract the substring before the first underscore
    char tmp_str[MAX_BUF_SIZE];
    sscanf(directory->d_name, "%*[^_]%*c%[^_]", tmp_str);

    // extract the substring after the first underscore
    char* token = strtok(NULL, "_");
    if (token != NULL) {
      // check whether the remaining substring ends with "metrics.csv"
      if (strcmp(token, "metrics.csv") == 0) {
        // if so, add the filename to dir_content
        strcpy(dir_content[num_el++], directory->d_name);
      }
    }
  }

  // close the directory
  closedir(ptr);

  // return the number of matching files found
  return num_el;
}
*/


// read and assemble metrics to send
void get_tx_string(char **send_metrics, int lines_to_read) {

  int curr_pos = 0;
  /*
  char dir_content[1][MAX_BUF_SIZE];
  int dir_el;
  dir_el = getDirContent(METRICS_DIR, dir_content);
  */
  int dir_el = 1;
  char *dir_content = "1010123456002_metrics.csv";
  char *metrics_string = "";
  for (int i = 0; i < dir_el; ++i) {
    // assemble path of file to read
    char file_path[MAX_BUF_SIZE] = METRICS_DIR;
    strcat(file_path, dir_content);

    // read metrics, always skip header
    readLastMetricsLines(file_path, lines_to_read, &metrics_string, 1);

    if (strlen(metrics_string) > 1) {
      int metrics_size = strlen(metrics_string);

      if (!(*send_metrics)) {
        *send_metrics = (char*) calloc(metrics_size, sizeof(char*));
        strcpy(*send_metrics, metrics_string);
      }
      else {
        *send_metrics = (char*) realloc(*send_metrics, (strlen(*send_metrics) + metrics_size) * sizeof(char*));
        memset(*send_metrics + curr_pos, '\0', metrics_size * sizeof(char*));

        strcat(*send_metrics, metrics_string);
      }

      curr_pos += metrics_size;

      free(metrics_string);
      metrics_string = NULL;
    }
  }
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


// remove substring from string
void remove_substr (char *string, char *sub) {
    char *match;
    int len = strlen(sub);
    while ((match = strstr(string, sub))) {
        *match = '\0';
        strcat(string, match+len);
    }
}

// tester function
int csv_tester(void) {

  char *send_metrics = NULL;
  int lines_to_read = 1; // 2 (original value)

  get_tx_string(&send_metrics, lines_to_read);

  if (send_metrics) {
    printf("len %d\n%s", strlen(send_metrics), send_metrics);

    // split if more than maximum payload for ric indication report
    if (strlen(send_metrics) > MAX_REPORT_PAYLOAD) {
      char *tmp_buf = NULL;
      tmp_buf = (char*) calloc(MAX_REPORT_PAYLOAD + 1, sizeof(char));

      // send in chunks, append
      for (int i = 0; i < strlen(send_metrics); i += MAX_REPORT_PAYLOAD) {
        memset(tmp_buf, 0, MAX_REPORT_PAYLOAD + 1);

        // add 'm' at the beginning to indicate there are more chunks
        int offset = 0;
        if (i + MAX_REPORT_PAYLOAD < strlen(send_metrics)) {
          strcpy(tmp_buf, "m");
          offset = 1;
        }

        strncpy(tmp_buf + offset, send_metrics + i, MAX_REPORT_PAYLOAD);

        printf("Chunk\n%s\n\n", tmp_buf);
      }

      free(tmp_buf);
      tmp_buf = NULL;
    }
  }

  return 0;
}
