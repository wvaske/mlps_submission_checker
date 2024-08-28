#!/usr/bin/env python3
#
# Copyright (C) 2024 Micron Technology, Inc. All rights reserved
#
# 28-Aug-2024  Wes Vaske

import json
import os
import sys

from datetime import datetime
from statistics import mean, stdev


workloads = ["unet", "resnet", "cosmoflow"]
gpus = ['a100', 'h100']

CACHE_PERCENT_THRESHOLD = 0.2


class MLPerfStorageResults:

    def __init__(self, storage_root_dir=None):

        # Parameters need tob e set for check_init to work
        self.storage_root_dir = storage_root_dir
        self.check_init()

        # Logic after knowing that init inputs are valid
        self.submitters = os.listdir(storage_root_dir)
        non_submitter_dirs = ["CONTRIBUTING.md", ".github", "README.md", "LICENSE.md"]
        self.submitters = [s for s in self.submitters if s not in non_submitter_dirs]

        # The individual dicts hold "report_path":["list of summary.json paths"]
        self.submitter_report_summary_map = {submitter: dict() for submitter in self.submitters}

        # The dicts hold <issue_type>: {<report path>:<detail_dict>}
        self.submitter_issues = {submitter: dict() for submitter in self.submitters}

        self.generate_run_mapping()

    def check_init(self):
        messages = []
        if self.storage_root_dir is None:
            messages.append(f'Storage root directory is not provided. '
                            f'Please provide the root directory of an MLPerf Storage Submission')

        if messages:
            for message in messages:
                print("ERROR - {message}")
            print('Failed to initialize MLPerfStorageResults object')
            raise Exception("Configuration Error")

    def generate_run_mapping(self):
        for submitter in self.submitters:
            submitter_dir = os.path.join(self.storage_root_dir, submitter)
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

    def check_time_between_runs(self):
        for submitter in self.submitters:
            submitter_flagged = False
            submitter_run_mapping = self.submitter_report_summary_map[submitter]
            for report, summaries in submitter_run_mapping.items():
                times = list()
                AUs = list()
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

                if any(delta > mean_run_time for delta in deltas):
                    issue_details = {report:
                                         {"deltas": ", ".join([f"{delta}" for delta in deltas]),
                                          "run_times": ", ".join([f"{rt}" for rt in run_times]),
                                          "aus": ", ".join([f"{au}%" for au in AUs]),
                                          "au_std": round(stdev(AUs), 4)
                                          }
                                     }

                    if not submitter_flagged:
                        self.submitter_issues[submitter] = {"time_between_runs": dict()}
                        submitter_flagged = True
                    self.submitter_issues[submitter]["time_between_runs"].update(issue_details)

    def check_file_system_caching(self):
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

                if len(cached_runs) == 0:
                    continue

                issue_details = {
                        report: {
                            "cached_percents": ", ".join([f"{round(cp*100, 2)}%" for cp in cached_percent]),
                            "file_cache_size (MiB)": "; ".join([f"{fc:,}" for fc in file_cache]),
                            "aus": ", ".join([f"{au}%" for au in AUs]),
                            "au_std": round(stdev(AUs), 4)
                        }
                    }

                if len(cached_runs) > 0 and not submitter_flagged:
                    submitter_flagged = True
                    self.submitter_issues[submitter] = dict()

                if len(cached_runs) == 1:
                    if "single_cached_run" not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter]["single_cached_run"] = dict()

                    self.submitter_issues[submitter]["single_cached_run"].update(issue_details)

                if len(cached_runs) > 1:
                    if "multiple_cached_runs" not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter]["multiple_cached_runs"] = dict()

                    self.submitter_issues[submitter]["multiple_cached_runs"].update(issue_details)

    def print_report(self, print_detail=False):
        for submitter, issues in self.submitter_issues.items():
            print(f'\nSubmitter: {submitter}')
            for issue_type, details in issues.items():
                print(f'  Issue Type: {issue_type}')
                for report, issue_detail in details.items():
                    print(f'    Report: {os.path.dirname(report)}')
                    if print_detail:
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


def main(dir_to_walk):
    mlperf_storage_results = MLPerfStorageResults(storage_root_dir=dir_to_walk)
    mlperf_storage_results.check_time_between_runs()
    mlperf_storage_results.check_file_system_caching()

    mlperf_storage_results.print_report(print_detail=True)


if __name__ == "__main__":
    root_path = "." if len(sys.argv) == 1 else sys.argv[1]
    main(root_path)
