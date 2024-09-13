#!/usr/bin/env python3
#
# Copyright (C) 2024 Micron Technology, Inc. All rights reserved
#
# 28-Aug-2024  Wes Vaske

import argparse
import json
import os
import subprocess

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
    parser.add_argument('-b', '--benchmark-script', default="/root/storage/",
                        help="The directory containing 'benchmark.sh'. Used for the dataset sizing checks")
    parser.add_argument('-s', '--filter-submitters', nargs="+", default=[],
                        help="Space separated list of submitters to check")
    parser.add_argument('-x', '--exclude-submitters', nargs="+", default=[],
                        help="List of submitters to exclude")
    parser.add_argument('-p', '--perf-reports', action="store_true",
                        help="Print the performance report for each report")
    parser.add_argument('-c', '--print-csv', action="store_true",
                        help="Print CSV of performance data")
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
    parser.add_argument('--check-checkpoint-files-in-code', action="store_true",
                        help="Check for checkpoint files in code.")
    parser.add_argument('--check-num-epochs', action="store_true",
                        help="Verify that each run did 5 epochs")
    parser.add_argument('--check-reports-exist', action="store_true",
                        help="Verify the reportgen has been run for every run")
    parser.add_argument('--check-dataset-size', action='store_true',
                        help="Verify the dataset size was above the minimum requirement")

    return {k: v for k, v in vars(parser.parse_args()).items()}


def get_datasize(benchmark_bin, workload, accelerator_type, num_accelerators, num_hosts, client_host_memory_in_gb):
    cmd = f"{benchmark_bin} datasize -w {workload} -g {accelerator_type} -n {num_accelerators} -c {num_hosts}" \
          f" -m {client_host_memory_in_gb}"
    completed_command = subprocess.run(cmd, shell=True, capture_output=True)
    output = completed_command.stdout.splitlines()
    datasize_files = output[1].split()[1].decode()
    return int(datasize_files)


class MLPerfStorageResults:

    def __init__(self, root_path=None, verbose=False, check_all=False, reports_only=False, filter_submitters=None,
                 benchmark_script=None, perf_reports=False, exclude_submitters=None, print_csv=None, *args, **kwargs):

        # Parameters need tob e set for check_init to work
        self.root_path = root_path
        self.verbose = verbose
        self.check_all = check_all
        self.reports_only = reports_only
        self.perf_reports = perf_reports
        self.print_csv = print_csv
        self.benchmark_script_dir = benchmark_script
        self.benchmark_script_bin = os.path.join(benchmark_script, "benchmark.sh")

        self.checks = {k: v for k, v in kwargs.items() if k.startswith("check_")}
        for key in self.checks.keys():
            if key.startswith('check_'):
                kwargs.pop(key)

        self._kwargs = kwargs
        self.validate_init()

        # Logic after knowing that init inputs are valid
        self.submitters = os.listdir(root_path)
        print(f'Processing {len(self.submitters)} submitters...')
        non_submitter_dirs = ["CONTRIBUTING.md", ".github", "README.md", "LICENSE.md", ".DS_Store"]
        self.submitters = [s for s in self.submitters if s not in non_submitter_dirs]

        if filter_submitters:
            filter_submitters = [s.lower() for s in filter_submitters]
            self.submitters = [s for s in self.submitters if s.lower() in filter_submitters]

        if exclude_submitters:
            exclude_submitters = [s.lower() for s in exclude_submitters]
            self.submitters = [s for s in self.submitters if s.lower() not in exclude_submitters]

        # The individual dicts hold "report_path":["list of summary.json paths"]
        self.submitter_report_summary_map = {submitter: dict() for submitter in self.submitters}
        self.submitter_report_cgroup_map = {submitter: dict() for submitter in self.submitters}
        self.submitter_performance_map = {submitter: dict() for submitter in self.submitters}

        # The dicts hold <issue_type>: {<report path>:<detail_dict>}
        self.submitter_issues = {submitter: dict() for submitter in self.submitters}

        self.submitter_file_list = dict()
        self.checks_run = []

        self.generate_run_mapping()
        self.run_checks()

        if self.perf_reports or self.print_csv:
            self.generate_performance_report()

        if self.print_csv:
            self.print_csv_data()
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
        num_summaries = 0
        num_reports = 0
        for submitter in self.submitters:
            submitter_dir = os.path.join(self.root_path, submitter)
            if not os.path.isdir(submitter_dir):
                print(f'WARNING - Submitter directory {submitter_dir} does not exist')
                continue

            submitter_file_list = get_file_list(submitter_dir)
            submitter_summaries = [f for f in submitter_file_list if os.path.basename(f) == "summary.json" if "code" not in f]
            submitter_reports = [f for f in submitter_file_list if os.path.basename(f) == "mlperf_storage_report.json" if "code" not in f]
            submitter_cgroups = [f for f in submitter_file_list if os.path.basename(f) == "cgroups.json" if "code" not in f]

            self.submitter_file_list[submitter] = submitter_file_list
            num_summaries += len(submitter_summaries)
            num_reports += len(submitter_reports)

            # For a given report, there should be a number of summary.json files that correspond with the same base path
            for report in submitter_reports:
                report_base_path = os.path.dirname(report)
                report_summaries = [f for f in submitter_summaries if f.startswith(report_base_path)]
                report_cgroups = list()

                report_path_up = report_base_path
                while report_path_up:
                    report_cgroups = [f for f in submitter_cgroups if f.startswith(report_path_up)]
                    if report_cgroups:
                        break
                    else:
                        report_path_up = os.path.dirname(report_path_up)

                if len(report_cgroups) > 1:
                    print(f'Have more than 1 cgroups.json file for a given report. '
                          f'\n\tReport: {report}\n\t\t{report_cgroups}')

                if report_cgroups:
                    report_cgroups = report_cgroups[0]
                else:
                    report_cgroups = None

                self.submitter_report_summary_map[submitter][report] = report_summaries
                self.submitter_report_cgroup_map[submitter][report] = report_cgroups

        print(f'Processing {num_reports} Reports and {num_summaries} Summaries')

    def generate_performance_report(self):
        for submitter in self.submitters:
            for report, summaries in self.submitter_report_summary_map[submitter].items():
                report_result_dict = dict(
                    au_list=list()
                )
                for summary in summaries:
                    with open(summary, 'r') as summary_file:
                        summary_dict = json.load(summary_file)

                    try:
                        report_result_dict["au_list"].append(round(summary_dict['metric']['train_au_mean_percentage'], 2))
                    except Exception as e:
                        import pdb
                        pdb.set_trace()

                report_result_dict['mean_au'] = mean([round(au, 2) for au in report_result_dict["au_list"]])

                with open(report, 'r') as report_file:
                    report_dict = json.load(report_file)

                report_result_dict['workload'] = report_dict['overall']['model']
                report_result_dict['accelerator'] = report_dict['overall']['accelerator']

                for k, v in report_dict['overall'].items():
                    if k.startswith("train_"):
                        report_result_dict[k] = round(float(v), 2)

                self.submitter_performance_map[submitter][report] = report_result_dict

    def add_issue_details(self, submitter, issue_key, issue_details, report=None):
        if issue_key not in self.submitter_issues[submitter].keys():
            self.submitter_issues[submitter][issue_key] = dict()

        if report:
            if report not in self.submitter_issues[submitter][issue_key].keys():
                self.submitter_issues[submitter][issue_key][report] = dict()

            self.submitter_issues[submitter][issue_key][report].update(issue_details)

        elif not report:
            self.submitter_issues[submitter][issue_key].update(issue_details)

    @staticmethod
    def get_summary_config_dict(summary):
        # We are looking for a config.yaml and overrides.yaml that have the same basepath as the summary.json
        base_dir = os.path.dirname(summary)
        config_yaml = os.path.join(base_dir, "configs", "config.yaml")
        overrides_yaml = os.path.join(base_dir, "configs", "overrides.yaml")

        if not os.path.isfile(config_yaml):
            print(f'Summary does not have associated config: {summary}')
        if not os.path.isfile(overrides_yaml):
            print(f"Summary does not have associated overrides: {summary}")

        # We will start by reading the first line in overrides which should give us the info we need. We'll use other
        # methods if necessary
        with open(overrides_yaml) as open_overrides:
            data = open_overrides.readlines()

        if data[0].startswith("- workload="):
            workload, accelerator = data[0].split("=")[1].strip().split('_')

            return dict(workload=workload, accelerator=accelerator)

        else:
            print(f'Unable to determine workload and accelerator from overrides for summary: {summary}')

    def run_checks(self):
        for check, enabled in self.checks.items():
            if (self.check_all or enabled) and hasattr(self, check):
                print(f'Running Check: {check}')
                self.checks_run.append(check)
                getattr(self, check)()

    def check_reports_exist(self):
        NON_ISSUE = "NON-ISSUE: reports_exist_for_all_summaries"
        SUMMARIES_WITHOUT_REPORTS_ISSUE = "ISSUE: summaries_without_report"

        # Whereever there is a summary file, there should be a report 1 level up
        # We'll generate a flat list of the reports and summaries
        # Then verify that summaries are matched to ../report
        for submitter in self.submitters:
            summary_files = [f for f in self.submitter_file_list[submitter]
                             if os.path.basename(f) == "summary.json"]

            summary_report_map = dict()
            for sf in summary_files:
                if "code" in sf:
                    continue

                sf_dir = os.path.dirname(sf)
                report_dir = os.path.dirname(sf_dir)

                report_dir_files = os.listdir(report_dir)
                report_files = [f for f in report_dir_files if os.path.basename(f) == "mlperf_storage_report.json"]

                if not report_files:
                    up_report_dir = os.path.dirname(report_dir)
                    up_report_dir_files = os.listdir(up_report_dir)
                    up_report_files = [f for f in up_report_dir_files if os.path.basename(f) == "mlperf_storage_report.json"]
                    report_files.extend(up_report_files)

                    if not report_files:
                        up_up_report_dir = os.path.dirname(up_report_dir)
                        up_up_report_dir_files = os.listdir(up_up_report_dir)
                        up_up_report_files = [f for f in up_up_report_dir_files if os.path.basename(f) == "mlperf_storage_report.json"]
                        report_files.extend(up_up_report_files)

                if len(report_files) > 1:
                    print(f'Have more than 1 report files?: {sf} \n{report_files}')

                if len(report_files) == 1:
                    report_file = report_files[0]
                    summary_report_map[sf] = report_file

                if not report_files:
                    summary_report_map[sf] = None

            issue_details = {"all_reports_generated": {
                "summaries_with_no_report": f"\n{35*' '}".join([sf for sf, report in summary_report_map.items() if report is None])
            }}

            non_issue_details = {"all_reports_generated": {
                "summaries_with_reports": f"\n{32*' '}".join([sf for sf, report in summary_report_map.items() if report])
            }}

            if non_issue_details["all_reports_generated"]["summaries_with_reports"] and self.verbose:
                self.add_issue_details(submitter=submitter,
                                       issue_key="NON-ISSUE: All summaries have a generated report",
                                       issue_details=non_issue_details)

            if issue_details['all_reports_generated']['summaries_with_no_report']:
                self.add_issue_details(submitter=submitter,
                                       issue_key="ISSUE: Summaries_without_corresponding_report",
                                       issue_details=issue_details)

    def check_dataset_size(self):
        NON_ISSUE_CORRECT = "NON-ISSUE: correct_dataset_size"
        NON_ISSUE_LARGER = "NON-ISSUE: dataset_larger_than_necessary"
        DATASET_TOO_SMALL_ISSUE = "ISSUE: dataset_too_small"

        # We will need to be able to run the 'benchmark datasize' tool based on workload, accelerator model and count
        # this will be fun
        for submitter in self.submitters:
            print(f'Checking submitter dataset size: {submitter}')
            for report, summaries in self.submitter_report_summary_map[submitter].items():
                datasize_minimums = []
                num_files_trains = []
                as_percents = []
                cgroup_memory_in_gb = None

                if cgroup_file := self.submitter_report_cgroup_map[submitter].get(report):
                    with open(cgroup_file) as open_cgroup_file:
                        cgroup_dict = json.load(open_cgroup_file)

                    cgroup_memory_in_gb = int(int(cgroup_dict['mem_limit_bytes']/1024/1024/1024))

                for summary in summaries:
                    if "code" in summary:
                        continue
                    with open(summary) as open_summary:
                        sum_data = json.load(open_summary)

                    sum_config_dict = self.get_summary_config_dict(summary=summary)
                    client_host_memory_in_gb = int(int(sum_data['host_meminfo']['MemTotal'].split()[0])/1024/1024)

                    if cgroup_memory_in_gb:
                        client_host_memory_in_gb = cgroup_memory_in_gb

                    run_dict = dict(
                        benchmark_bin=self.benchmark_script_bin,
                        workload=sum_config_dict.get('workload'),
                        accelerator_type=sum_config_dict.get('accelerator'),
                        num_accelerators=sum_data['num_accelerators'],
                        num_hosts=sum_data['num_hosts'],
                        client_host_memory_in_gb=client_host_memory_in_gb
                    )

                    datasize_minimums.append(get_datasize(**run_dict))
                    num_files_trains.append(sum_data['num_files_train'])
                    as_percents.append(round((num_files_trains[-1] / datasize_minimums[-1])*100, 1))

                issue_details = {report:

                                     {
                                         "workload": sum_config_dict.get('workload'),
                                         'accelerator': sum_config_dict.get('accelerator'),
                                         "num_accelerators": sum_data['num_accelerators'],
                                         "num_hosts": sum_data['num_hosts'],
                                         "client_host_memory_in_gb": client_host_memory_in_gb,
                                         "datasize_minimums": datasize_minimums,
                                         "num_files_train": num_files_trains,
                                         "train_as_percent_of_requirement": as_percents}
                                 }

                if any([asp < 100 for asp in as_percents]):
                    self.add_issue_details(submitter=submitter,
                                           issue_key=DATASET_TOO_SMALL_ISSUE,
                                           issue_details=issue_details)

                elif all([100 <= asp <= 105 for asp in as_percents]) and self.verbose:
                    self.add_issue_details(submitter=submitter,
                                           issue_key=NON_ISSUE_CORRECT,
                                           issue_details=issue_details)
                elif any([asp > 105 for asp in as_percents]) and self.verbose:
                    self.add_issue_details(submitter=submitter,
                                           issue_key=NON_ISSUE_LARGER,
                                           issue_details=issue_details)

    def check_checkpoint_files_in_code(self):
        NON_ISSUE = "NON-ISSUE: checkpoint_files_not_in_code"
        CHECKPOINT_FILES_IN_CODE_ISSUE = "ISSUE: checkpoint_file_in_code"

        for submitter in self.submitters:
            submitter_files = get_file_list(os.path.join(self.root_path, submitter))
            checkpoint_files = [f for f in submitter_files
                                if os.path.basename(f).startswith("layer")
                                and f.endswith(".pt")
                                and "code" in f]

            issue_details = {"No-specific-report": {"checkpoint_files": checkpoint_files}}
            if checkpoint_files:
                self.add_issue_details(submitter=submitter,
                                       issue_key=CHECKPOINT_FILES_IN_CODE_ISSUE,
                                       issue_details=issue_details)

            elif self.verbose:
                self.add_issue_details(submitter=submitter,
                                       issue_key=NON_ISSUE,
                                       issue_details=issue_details)

    def check_num_epochs(self):
        NON_ISSUE = "NON-ISSUE: runs_are_5_epochs"
        WRONG_EPOCH_COUNT_ISSUE = "ISSUE: wrong_epoch_count"

        for submitter in self.submitters:
            for report, summaries in self.submitter_report_summary_map[submitter].items():
                epochs = []
                for summary in summaries:
                    with open(summary) as open_summary:
                        sum_data = json.load(open_summary)

                    epochs.append(sum_data["epochs"])
                    # print(epochs)

                    # We can have multiple issues per report so we have 1 more key here than for other checks
                issue_details = {report:
                                     {"epochs": epochs
                                      }
                                 }

                if all([e == 5 for e in epochs]) and self.verbose:
                    self.add_issue_details(submitter=submitter,
                                           issue_key=NON_ISSUE,
                                           issue_details=issue_details)

                if any([e != 5 for e in epochs]):
                    self.add_issue_details(submitter=submitter,
                                           issue_key=WRONG_EPOCH_COUNT_ISSUE,
                                           issue_details=issue_details)

    def check_inter_test_times(self):
        NON_ISSUE = "NON-ISSUE: minimal_time_between_runs"
        TIME_BETWEEN_RUNS_ISSUE = "NON-ISSUE: time_between_runs_exceeds_avg_runtime"

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

                issue_details = {report:
                                     {"deltas": ", ".join([f"{delta}" for delta in deltas]),
                                      "run_times": ", ".join([f"{rt}" for rt in run_times]),
                                      "aus": ", ".join([f"{au}%" for au in AUs]),
                                      "au_std": round(stdev(AUs), 4)
                                      }
                                 }

                if self.verbose:
                    if any(delta > mean_run_time for delta in deltas):
                        if TIME_BETWEEN_RUNS_ISSUE not in self.submitter_issues[submitter].keys():
                            self.submitter_issues[submitter][TIME_BETWEEN_RUNS_ISSUE] = dict()
                        self.submitter_issues[submitter][TIME_BETWEEN_RUNS_ISSUE].update(issue_details)

                    if all(delta < mean_run_time for delta in deltas):
                        if NON_ISSUE not in self.submitter_issues[submitter].keys():
                            self.submitter_issues[submitter][NON_ISSUE] = dict()
                        self.submitter_issues[submitter][NON_ISSUE].update(issue_details)

    def check_file_system_caching(self):
        NON_ISSUE = 'NON-ISSUE: no_cached_runs'
        SINGLE_CACHED_RUN_ISSUE = "NON-ISSUE: single_cached_run"
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
                    if "code" in summary:
                        continue

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

                if len(AUs) <= 1:
                    print(f'Not checking for filesystem caching, code directory?: ')

                # Convert to MiB
                mem_total = [int(int(m.split(' ')[0])/1024) for m in mem_total]
                cached = [int(int(c.split(' ')[0])/1024) for c in cached]
                active_file = [int(int(a.split(' ')[0])/1024) for a in active_file]
                inactive_file = [int(int(i.split(' ')[0])/1024) for i in inactive_file]

                file_cache = [active_file[i] + inactive_file[i] for i in range(len(mem_total))]
                cached_percent = [round((active_file[i] + inactive_file[i]) / mem_total[i], 4) for i in range(len(mem_total))]
                cached_runs = [cp for cp in cached_percent if cp > CACHE_PERCENT_THRESHOLD]

                try:
                    issue_details = {
                            report: {
                                "cached_percents": ", ".join([f"{round(cp*100, 2)}%" for cp in cached_percent]),
                                "file_cache_size (MiB)": "; ".join([f"{fc:,}" for fc in file_cache]),
                                "aus": ", ".join([f"{au}%" for au in AUs]),
                                "au_std": round(stdev(AUs), 4)
                            }
                        }
                except Exception as e:
                    import pdb
                    pdb.set_trace()

                if not cached_runs and self.verbose:
                    if NON_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][NON_ISSUE] = dict()
                    self.submitter_issues[submitter][NON_ISSUE].update(issue_details)

                if len(cached_runs) == 1 and self.verbose:
                    if SINGLE_CACHED_RUN_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][SINGLE_CACHED_RUN_ISSUE] = dict()
                    self.submitter_issues[submitter][SINGLE_CACHED_RUN_ISSUE].update(issue_details)

                if len(cached_runs) > 1:
                    if MULTIPLE_CACHED_RUNS_ISSUE not in self.submitter_issues[submitter].keys():
                        self.submitter_issues[submitter][MULTIPLE_CACHED_RUNS_ISSUE] = dict()
                    self.submitter_issues[submitter][MULTIPLE_CACHED_RUNS_ISSUE].update(issue_details)

    def print_report(self):
        if self.checks_run:
            for submitter, issues in self.submitter_issues.items():
                print(f'\nSubmitter: {submitter}')
                if not issues:
                    print(f'  No issues found')
                    continue
                for issue_type, details in issues.items():
                    print(f'  {issue_type}')
                    for report, issue_detail in details.items():
                        print(f'    Report: {os.path.dirname(report)}')
                        if not self.reports_only:
                            if hasattr(list(issue_detail.values())[0], "keys"):
                                # Go down one more layer

                                for summary, summary_details in issue_detail.items():
                                    print(f'      Summary: {summary}:')
                                    for key, value in summary_details.items():
                                        print(f'        {key}: {value}')

                                print()
                            else:
                                print(f'      Details:')
                                for key, value in issue_detail.items():
                                    print(f'        {key}: {value}')
                    print()

        if self.perf_reports:
            for submitter, report_dict in self.submitter_performance_map.items():
                print(f'\nSubmitter: {submitter}')
                for report, report_details in report_dict.items():
                    print(f'  Report: {report}')
                    for k, v in report_details.items():
                        print(f'    {k}: {v}')
                    print()
                print()

    def print_csv_data(self):
        csv_data = []  # list of dictionaries
        for submitter, report_dict in self.submitter_performance_map.items():
            for report, report_details in report_dict.items():
                tmp_dict = report_details.copy()
                tmp_dict['report'] = os.path.dirname(report)
                tmp_dict['submitter'] = submitter
                del tmp_dict['au_list']
                csv_data.append(tmp_dict)

        import csv
        all_fields = set()
        for item in csv_data:
            # import pdb
            # pdb.set_trace()
            all_fields.update(item.keys())

        header_row = ""
        for field in all_fields:
            header_row += field + ","
        print(header_row)

        for item in csv_data:
            str_to_print = ""
            for field in all_fields:
                str_to_print += str(item.get(field, "")) + ","

            print(str_to_print)


def get_file_list(top_directory="."):
    file_list = []
    for root, dirs, files in os.walk(top_directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


if __name__ == "__main__":
    input_kwargs = parse_arguments()
    mlperf_storage_results = MLPerfStorageResults(**input_kwargs)
