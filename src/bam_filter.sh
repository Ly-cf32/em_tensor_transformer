#!/bin/bash

# 使用samtools来处理bam文件和索引文件
# 这里假设bam文件和索引文件都已经存在
BAM_FILE=$1
BAM_INDEX=$2


# 使用bedtools来处理GTF文件
# 这里假设bedtools已经安装并可用
GTF_FILE=$3
OUT_PREFIX=$4

# 使用bedtools intersect来获取符合GTF文件中注释的reads
bedtools intersect -abam $BAM_FILE -b $GTF_FILE -wa -split > $OUT_PREFIX.filtered_reads.bam

# 对筛选出的reads进行排序和索引
samtools sort -@ 4 $OUT_PREFIX.filtered_reads.bam -o $OUT_PREFIX.filtered_reads_sorted.bam
samtools index $OUT_PREFIX.filtered_reads_sorted.bam


