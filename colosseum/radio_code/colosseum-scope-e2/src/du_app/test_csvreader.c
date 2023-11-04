#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUFSIZE 512

void append_uint_to_str(unsigned int num, char* str) {
    char temp[20];
    sprintf(temp, "%u", num);
    strcat(str, temp);
}

void append_string(char **dest, char *src) {
    size_t dest_len = strlen(*dest);
    size_t src_len = strlen(src);
    if (dest_len == 0) {
        *dest = malloc(src_len + 1);
    }else{
        *dest = realloc(*dest, dest_len + src_len + 1); // +1 for null-terminator

    }
    strcat(*dest, src);
}



int main() {
  unsigned int to_read = 5;
  char *file_name = "/home/mauro/Research/ORAN/repo/traffic_gen/colosseum/1010123456002_metrics.csv";
  char *cmd_prefix = "/home/mauro/Research/ORAN/repo/traffic_gen/colosseum/source_code/colosseum-scope-e2/src/du_app/readLastMetrics.o -csv=";
  char *cmd = (char*) malloc((strlen(cmd_prefix)+1)*sizeof(char));
  strcpy(cmd, cmd_prefix);

  printf("[mau] after cmd init\n");
  append_string(&cmd, file_name);
  append_string(&cmd, " -ltr=");
  append_uint_to_str((unsigned int) to_read, cmd);
  //char cmd[300] = "/home/mauro/Research/ORAN/repo/traffic_gen/colosseum/source_code/colosseum-scope-e2/src/du_app/readLastMetrics.o -csv=/home/mauro/Research/ORAN/repo/traffic_gen/colosseum/1010123456002_metrics.csv -ltr=";
  //append_uint_to_str(to_read, cmd);


    char *output_metrics = "";
    size_t valid_metrics = 0;
    char buf[BUFSIZE];
    FILE *fp;

    if ((fp = popen(cmd, "r")) == NULL) {
        printf("Error opening pipe!\n");
        return -1;
    }

    while (fgets(buf, BUFSIZE, fp) != NULL) {
      if (!(strcmp(buf, "\n") == 0)){
          append_string(&output_metrics, buf);
          valid_metrics++;
          printf("--> %s", buf);
      }

    }
    if (strlen(output_metrics) > 1){
        size_t num_chars = strlen(output_metrics);
        output_metrics[num_chars + 1] = '\0';
    }

    if (pclose(fp)) {
        printf("Command not found or exited with error status\n");
        return -1;
    }

    printf("Tot. Return output: %s", output_metrics);
    free(output_metrics);

    return 0;
}