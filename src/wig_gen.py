# bedgraph_to_wig.py
import sys

def bedgraph_to_wig(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        current_chrom = None
        for line in infile:
            chrom, start, end, score = line.strip().split('\t')
            if chrom != current_chrom:
                current_chrom = chrom
                outfile.write(f"variableStep chrom={current_chrom}\n")
            outfile.write(f"{start}\t{score}\n")

if __name__ == "__main__":
    input_bedgraph = sys.argv[1]
    output_wig = sys.argv[2]
    bedgraph_to_wig(input_bedgraph, output_wig)
