#!/usr/bin/python3
import cx_Oracle
import json
import logging
import os
import subprocess
import time

"""
Script that queries the database for massive jobs and, when finding one,
starts a cluster search.
"""

# Template for starting a script
run_template = """#!/bin/bash
#SBATCH --partition={}
#SBATCH -n 5
#SBATCH --time=1:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
# --------------------------------------------------------------------------

srun ./AI4S_runscript.sh"""


def rows_to_dict_list(cursor):
    """ Converts an Oracle return value into a list of dictionaries.

    Parameters
    ----------
    cursor : cx_Oracle.Cursor
        A cursor to a set of results

    Returns
    -------
    list(dict)
        A list of dictionaries, one for each row in the result set. Each
        dictionary contains as key the (lowercase) name of each column.

    Notes
    -----
    The short way of running this code is
    `return [dict(zip(map(str.lower, columns), row)) for row in cursor]`,
    but this code returns `LOB` instead of `string`. Rather than keep nesting
    maps inside maps to solve this problem, I unrolled the loops.
    """
    columns = [i[0].lower() for i in cursor.description]
    retval = []
    for row in cursor:
        new_dict = dict()
        for i in range(len(row)):
            value = row[i]
            if type(value) is cx_Oracle.LOB:
                new_dict[columns[i]] = value.read()
            else:
                new_dict[columns[i]] = value
        retval.append(new_dict)
    return retval


def is_running():
    is_process_running = False
    ps = subprocess.Popen(('squeue'), stdout=subprocess.PIPE)
    try:
        output = subprocess.check_output(('grep', 'gkdxc'), stdin=ps.stdout)
        ps.wait()
    except subprocess.CalledProcessError:
        # No results found
        is_process_running = False
    return is_process_running


def get_pending_jobs():
    pending_jobs = False
    queue = "long"
    # Open a connection
    host, port, service, username, password = "first_line.split(':')"
    conn_string = "{}:{}/{}".format(host, port, service)
    connection = cx_Oracle.connect(username, password, conn_string)

    # Query the number of jobs
    query_jobs = "SELECT queue, job_ids FROM massive_job ORDER BY ID desc"
    query_single_job = "SELECT status FROM job where id={}"
    cursor = connection.cursor()
    cursor.execute(query_jobs)
    row_dict = rows_to_dict_list(cursor)
    for row in row_dict:
        queue = row['queue']
        jobs = json.loads(row['job_ids'])
        for job in jobs:
            cursor.execute(query_single_job.format(job))
            single_row_dict = rows_to_dict_list(cursor)
            if len(single_row_dict) > 0:
                if single_row_dict[0]['status'] in {'pending', 'pending_massive'}:
                    pending_jobs = True
                    break
            if pending_jobs:
                break
    connection.close()
    return queue, pending_jobs


if __name__ == '__main__':
    script_location = './run_script.sh'
    queue = "long"
    logging.getLogger().setLevel(logging.INFO)
    while True:
        # Query the database
        queue, pending_jobs = get_pending_jobs()
        if pending_jobs:
            logging.info("There are pending jobs. Checking for running cluster")
            # There are pending jobs. But do I need to start a new search?
            if not is_running():
                logging.info("There are pending jobs and no running cluster. Starting a cluster run.")
                # There should be no running cluster job, so we make a new one.
                with open(script_location, 'w') as fp:
                    print(run_template.format(queue), file=fp)
                os.chmod(script_location, 0o755)
                rc = subprocess.call(script_location)
        else:
            logging.info("No pending jobs")
        # Wait 5 minutes before next check
        time.sleep(300)
