import argparse
import re
import subprocess

import pandas as pd
import psutil
import requests
import tabulate

kernel_regex = re.compile(r".+kernel-(.+)\.json")
notebook_regex = re.compile(r"(https?://([^:/]+):?(\d+)?)/?(\?token=([a-z0-9]+))?")


def get_proc_info():
    pids = psutil.pids()

    # memory info from psutil.Process
    df_mem = []

    for pid in pids:
        try:
            proc = psutil.Process(pid)
            cmd = " ".join(proc.cmdline())
        except psutil.NoSuchProcess:
            continue

        if len(cmd) > 0 and ("jupyter" in cmd or "ipython" in cmd) and "kernel" in cmd:
            # kernel
            kernel_ID = re.sub(kernel_regex, r"\1", cmd)

            # memory
            mem = proc.memory_info()[0] / float(1e9)

            uname = proc.username()

            # user, pid, memory, kernel_ID
            df_mem.append([uname, pid, mem, kernel_ID])

    df_mem = pd.DataFrame(df_mem)
    df_mem.columns = ["user", "pid", "memory_GB", "kernel_ID"]
    return df_mem


def get_running_notebooks():
    notebooks = []

    for n in subprocess.Popen(
        ["jupyter", "notebook", "list"], stdout=subprocess.PIPE
    ).stdout.readlines()[1:]:
        match = re.match(notebook_regex, n.decode())
        if match:
            base_url, host, port, _, token = match.groups()
            notebooks.append({"base_url": base_url, "token": token})
        else:
            print("Unknown format: {}".format(n.decode()))

    return notebooks


def get_session_info(password=None):
    df_nb = []
    kernels = []

    for notebook in get_running_notebooks():
        s = requests.Session()
        if notebook["token"] is not None:
            s.get(notebook["base_url"] + "/?token=" + notebook["token"])
        else:
            # do a get to the base url to get the session cookies
            s.get(notebook["base_url"])
        if password is not None:
            # Seems jupyter auth process has changed, need to first get a cookie,
            # then add that cookie to the data being sent over with the password
            data = {"password": password}
            data.update(s.cookies)
            s.post(notebook["base_url"] + "/login", data=data)

        res = s.get(notebook["base_url"] + "/api/sessions")

        if res.status_code != 200:
            raise Exception(res.json())

        for sess in res.json():
            kernel_ID = sess["kernel"]["id"]
            if kernel_ID not in kernels:
                kernel = {
                    "kernel_ID": kernel_ID,
                    "kernel_name": sess["kernel"]["name"],
                    "kernel_state": sess["kernel"]["execution_state"],
                    "kernel_connections": sess["kernel"]["connections"],
                    # "notebook_url": notebook["base_url"] + "/notebook/" + sess["id"],
                    "notebook_path": sess["path"],
                }
                kernel.update(notebook)
                df_nb.append(kernel)
                kernels.append(kernel_ID)

    df_nb = pd.DataFrame(df_nb)
    del df_nb["token"]
    return df_nb


def parse_args():
    parser = argparse.ArgumentParser(description="Find memory usage.")
    parser.add_argument("--password", help="password (only needed if pass-protected)")

    return parser.parse_args()


def main(password=None, print_ascii=False):
    df_mem = get_proc_info()
    df_nb = get_session_info(password)

    # joining tables
    df = pd.merge(df_nb, df_mem, on=["kernel_ID"], how="inner")
    df = df.sort_values("memory_GB", ascending=False).reset_index(drop=True)
    if print_ascii:
        print(tabulate.tabulate(df, headers=(df.columns.tolist())))
    return df


if __name__ == "__main__":
    main(None, print_ascii=True)
