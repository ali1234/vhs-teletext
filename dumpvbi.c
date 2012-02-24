/* dumpvbi.c :- quick hack to dump raw VBI samples from WinTV /dev/vbi
 *
 * Copyright 2011 Alistair Buxton <a.j.buxton@gmail.com>
 *
 * License: This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version. This program is distributed in the hope
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */

/*
 * Compile: gcc -o dumpvbi dumpvbi.c
 *
 * This program just reads raw samples from /dev/vbi0. It is functionally 
 * equivalent to 'cat /dev/vbi0' except that it splits the output into separate 
 * files, one for each frame.
 * 
 * It seems that the level of the signal drifts and sometimes the signal gets
 * clipped. I probably need to implement gain control on the BT chip to fix this.
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

typedef unsigned char u8;

/* This function just checks the format of the VBI device and prints it. */
int query_dev(int fd)
{
    struct v4l2_format v4l2_format[1];
    struct v4l2_vbi_format *vbifmt = &v4l2_format->fmt.vbi;

    v4l2_format->type = V4L2_BUF_TYPE_VBI_CAPTURE;

    if (ioctl(fd, VIDIOC_G_FMT, v4l2_format) == -1
	|| v4l2_format->type != V4L2_BUF_TYPE_VBI_CAPTURE) {
        goto err_ioctl;
    } else {
	int size;

        fprintf(stderr, "Device Info:\n");

	if(vbifmt->sample_format == V4L2_PIX_FMT_GREY)
            fprintf(stderr, "Pixel format: V4L2_PIX_FMT_GREY - OK\n");
	else
            fprintf(stderr, "Pixel format: Not recognised.\n");

        fprintf(stderr, "Samples per line: %d - %s\n", vbifmt->samples_per_line, ((vbifmt->samples_per_line==2048)?"OK":"BAD"));

        fprintf(stderr, "count[0]: %d - %s\n", vbifmt->count[0], ((vbifmt->count[0]==16)?"OK":"BAD"));
        fprintf(stderr, "count[1]: %d - %s\n", vbifmt->count[1], ((vbifmt->count[1]==16)?"OK":"BAD"));
    }

    return 0;

    err_ioctl:
        fprintf(stderr, "Couldn't get device info.\n");
        fprintf(stderr, "Continuing anyway, though this might not work.\n");
        return 0;

}


int main(int argc, char *argv[]) {

    int fd;
    FILE *f;
    char vbi_name[] = "/dev/vbi0";
    int c, n, err=0;
    char filename[20];

    u8 rawbuf[0x10000]; /* 2048*32 */

    if ((fd = open(vbi_name, O_RDONLY)) == -1) {
        fprintf(stderr, "Error opening %s.\n", vbi_name);
	goto err_file;
    }

    if (query_dev(fd) == -1) {
	goto err_query;
    }

    for(c = 0; ; c++) {
        n = read(fd, rawbuf, 0x10000);

        if (n != 0x10000) err++;
        else {
            sprintf(filename, "%08d.vbi", c);

            f = fopen(filename, "w");
            fwrite(rawbuf, n, 1, f);
            fclose(f);

            printf("%s - %d\r", filename, err);
        }
    }

    err_query:
        close(fd);
    err_file:
        ;
}
