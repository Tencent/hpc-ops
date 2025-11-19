import sys


def get_error_number(line):
    istart = line.find("(")
    iend = line.find(")")
    text = line[istart + 1 : iend]
    num_errors = int(text.split(",")[0].strip("errors").strip())
    return num_errors


if __name__ == "__main__":

    # demo case
    # ========= RACECHECK SUMMARY: 16 hazards displayed (16 errors, 0 warnings)

    for line in sys.stdin:
        print("{}".format(line), end="", flush=True)

        is_summary_line = "RACECHECK SUMMARY:" in line

        if is_summary_line and get_error_number(line) > 0:
            sys.exit(1)

    sys.exit(0)
