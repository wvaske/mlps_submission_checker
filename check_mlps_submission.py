#!/usr/bin/env python3
#
# Copyright (C) 2024 Micron Technology, Inc. All rights reserved
#
# 28-Aug-2024  Wes Vaske

import argparse
import json
import os
import sys

from datetime import datetime
from statistics import mean, stdev


workloads = ["unet", "resnet", "cosmoflow"]
gpus = ['a100', 'h100']

CACHE_PERCENT_THRESHOLD = 0.2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to check for specific issues in MLPerf Storage v1.0 submissions"
    )
    parser.add_argument('-r', '--root-path', required=True, type=str,
                        help="The root path of the submission directory. Top level directories in this path are "
                             "assumed to be the submitters")
    parser.add_argument('-s', '--filter-submitters', nargs="+", default=[],
                        help="Space separated list of submitters to check")
    parser.add_argument('-a', '--check-all', action="store_true",
                        help="Perform all checks")
    parser.add_argument('--reports-only', action="store_true",
                        help="Print only the reports with issues, not the details of the issues.")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="List details of non-issue reports as well as issue reports")
    parser.add_argument('--check-inter-test-times', action="store_true",
                        help="Analyze the time between runs for possible non-consecutive tests.")
    parser.add_argument('--check-file-system-caching', action="store_true",
                        help="Analyze the cache percentages to detect possible issues.")

    return {k: v for k, v in vars(parser.parse_args()).items()}


class MLPerfStorageResults:

    def __init__(self, root_path=None, verbose=False, check_all=False, reports_only=False, filter_submitters=None,
                 *args, **kwargs):

        # Parameters need tob e set for check_init to work
        self.root_path = root_path
        self.verbose = verbose
        self.check_all = check_all
        self.reports_only = reports_only

        self.checks = {k: v for k, v in kwargs.items() if k.startswith("check_")}
        for key in self.checks.keys():
            if key.startswith('check_'):
                kwargs.pop(key)

        self._kwargs = kwargs
        self.validate_init()

        # Logic after knowing that init inputs are valid
        self.submitters = os.listdir(root_path)
        non_submitter_dirs = ["CONTRIBUTING.md", ".github", "README.md", "LICENSE.md"]
        self.submitters = [s for s in self.submitters if s not in non_submitter_dirs]

        if filter_submitters:
            filter_submitters = [s.lower() for s in filter_submitters]
            self.submitters = [s for s in self.submitters if s.lower() in filter_submitters]

        # The individual dicts hold "report_path":["list of summary.json paths"]
        self.submitter_report_summary_map = {submitter: dict() for submitter in self.submitters}

        # The dicts hold <issue_type>: {<report path>:<detail_dict>}
        self.submitter_issues = {submitter: dict() for submitter in self.submitters}

        self.generate_run_mapping()
        self.run_checks()
        self.print_report()

    def validate_init(self):
        messages = []
        if self.root_path is None:
            messages.append(f'Root directory is not provided. '
                            f'Please provide the root directory of an MLPerf Storage Submission')

        if self._kwargs:
            for key, value in self._kwargs.items():
                messages.append(f'Unused input kwargs: {key}:{value}')

        if messages:
            for message in messages:
                print(f"ERROR - {message}")
            print('Failed to initialize MLPerfStorageResults object')
            raise Exception("Configuration Error")

    def generate_run_mapping(self):
        for submitter in self.submitters:
            submitter_dir = os.path.join(self.root_path, submitter)
            if not os.path.isdir(submitter_dir):
                print(f'WARNING - Submitter directory {submitter_dir} does not exist')
                continue

            submitter_file_list = get_file_list(submitter_dir)
            submitter_summaries = [f for f in submitter_file_list if os.path.basename(f) == "summary.json"]
            submitter_reports = [f for f in submitter_file_list if os.path.basename(f) == "mlperf_storage_report.json"]

            # For a given report, there should be a number of summary.json files that correspond with the same base path
            for report in submitter_reports:
                report_base_path = os.path.dirname(report)
                report_summaries = [f for f in submitter_summaries if f.startswith(report_base_path)]
                self.submitter_report_summary_map[submitter][report] = report_summaries

    def run_checks(self):
        for check, enabled in self.checks.items():
            if (self.check_all or enabled) and hasattr(self, check):
                getattr(self, check)()

    def check_inter_test_times(self):
        NON_ISSUE = "NON-ISSUE: time_between_runs"
        TIME_BETWEEN_RUNS_ISSUE = "ISSUE (possible): time_between_runs"

        for submitter in self.submitters:
            submitter_run_mapping = self.submitter_report_summary_map[submitter]
            for report, summaries in submitter_run_mapping.items():
                times = list()
                AUs = list()

                if len(summaries) != 5:
                    # This occurs when we get the test files from the code dir. A proper mlperf_storage_report.json
                    #   can only be created with 5 runs and anything else will not generate the report so we are safe
                    #   to do a quick check like this and pop on out
                    if self.verbose:
                        print(f'Skipping report from {submitter} due to not having 5 summary files: {report}')
                    continue

                for summary in sorted(summaries):
                    with open(summary) as open_summary:
                        sum_data = json.load(open_summary)
                    start_time = datetime.fromisoformat(sum_data["start"])
                    end_time = datetime.fromisoformat(sum_data["end"])
                    times.append((start_time, end_time))
                    AUs.append(round(sum_data['metric']['train_au_mean_percentage'], 2))

                run_times = [(e-s).seconds for s, e in times]
                mean_run_time = mean(run_times)
                deltas = []
                for i in range(len(times) - 1):
                    deltas.append((times[i + 1][0] - times[i][1]).seconds)

                if len(AUs) == 1:
                    import pdb
                    pdb.set_trace()
                issue_details = {report:
                                     {"deltas": ", ".join([f"{delta}" for delta in deltas]),
                                      "run_times": ", ".join([f"{rt}" for rt in run_times]),
                                      "aus": ", ".join([f"{au}%" for au in AUs]),
                                      "au_std": round(stdev(AUs), 4)
                                      }
                                 }

                if any(delta > mean_run_time for delta in deltas):
                    if TIME_BETWEEN_RUNS_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][TIME_BETWEEN_RUNS_ISSUE] = dict()
                    self.submitter_issues[submitter][TIME_BETWEEN_RUNS_ISSUE].update(issue_details)

                elif self.verbose:
                    if NON_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][NON_ISSUE] = dict()
                    self.submitter_issues[submitter][NON_ISSUE].update(issue_details)

    def check_file_system_caching(self):
        NON_ISSUE = 'NON-ISSUE: cached_runs'
        SINGLE_CACHED_RUN_ISSUE = "ISSUE (possible): single_cached_run"
        MULTIPLE_CACHED_RUNS_ISSUE = "ISSUE: multiple_cached_runs"

        for submitter in self.submitters:
            submitter_flagged = False
            submitter_run_mapping = self.submitter_report_summary_map[submitter]
            for report, summaries in submitter_run_mapping.items():
                mem_total = []
                cached = []
                active_file = []
                inactive_file = []
                AUs = []
                for summary in summaries:
                    with open(summary) as open_summary:
                        sum_data = json.load(open_summary)

                    try:
                        mem_total.append(sum_data['host_meminfo']['MemTotal'])
                        cached.append(sum_data['host_meminfo']['Cached'])
                        active_file.append(sum_data['host_meminfo']['Active(file)'])
                        inactive_file.append(sum_data['host_meminfo']['Inactive(file)'])
                        AUs.append(round(sum_data['metric']['train_au_mean_percentage'], 2))
                    except Exception as e:
                        continue

                # Convert to MiB
                mem_total = [int(int(m.split(' ')[0])/1024) for m in mem_total]
                cached = [int(int(c.split(' ')[0])/1024) for c in cached]
                active_file = [int(int(a.split(' ')[0])/1024) for a in active_file]
                inactive_file = [int(int(i.split(' ')[0])/1024) for i in inactive_file]

                file_cache = [active_file[i] + inactive_file[i] for i in range(len(mem_total))]
                cached_percent = [round((active_file[i] + inactive_file[i]) / mem_total[i], 4) for i in range(len(mem_total))]
                cached_runs = [cp for cp in cached_percent if cp > CACHE_PERCENT_THRESHOLD]

                issue_details = {
                        report: {
                            "cached_percents": ", ".join([f"{round(cp*100, 2)}%" for cp in cached_percent]),
                            "file_cache_size (MiB)": "; ".join([f"{fc:,}" for fc in file_cache]),
                            "aus": ", ".join([f"{au}%" for au in AUs]),
                            "au_std": round(stdev(AUs), 4)
                        }
                    }

                if not cached_runs and self.verbose:
                    if NON_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][NON_ISSUE] = dict()
                    self.submitter_issues[submitter][NON_ISSUE].update(issue_details)

                if len(cached_runs) == 1:
                    if SINGLE_CACHED_RUN_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][SINGLE_CACHED_RUN_ISSUE] = dict()
                    self.submitter_issues[submitter][SINGLE_CACHED_RUN_ISSUE].update(issue_details)

                if len(cached_runs) > 1:
                    if MULTIPLE_CACHED_RUNS_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][MULTIPLE_CACHED_RUNS_ISSUE] = dict()
                    self.submitter_issues[submitter][MULTIPLE_CACHED_RUNS_ISSUE].update(issue_details)

    def print_report(self):
        for submitter, issues in self.submitter_issues.items():
            print(f'\nSubmitter: {submitter}')
            for issue_type, details in issues.items():
                print(f'  {issue_type}')
                for report, issue_detail in details.items():
                    print(f'    Report: {os.path.dirname(report)}')
                    if not self.reports_only:
                        print(f'      Details:')
                        for key, value in issue_detail.items():
                            print(f'        {key}: {value}')
                print()


def get_file_list(top_directory="."):
    file_list = []
    for root, dirs, files in os.walk(top_directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


if __name__ == "__main__":
    input_kwargs = parse_arguments()
    mlperf_storage_results = MLPerfStorageResults(**input_kwargs)
