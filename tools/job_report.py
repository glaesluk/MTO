import argparse
import logging
import os
from db_utils import DBConnection
from tools.report import plot_routes, write_job_report
from utils import get_routes_and_metadata_from_job

if __name__ == '__main__':
    # Read the command line arguments
    parser = argparse.ArgumentParser(description='Generates a report for a single job')
    parser.add_argument('job_id', type=int, nargs='+',
                        help='ID of the job(s) for the report(s)')
    parser.add_argument('--outdir', type=str,
                        help='Path where the report(s) will be stored')
    args = parser.parse_args()

    # Connect to the database
    dbfile = "/etc/db-access/compute_app_db_credentials"
    with open(dbfile, "r") as fp:
        first_line = fp.readline().strip()
    host, port, service, username, password = first_line.split(':')
    conn_string = "{}:{}/{}".format(host, port, service)
    connection = DBConnection(conn_string, username, password, db_type="oracle")
    connection.connect()

    # Collect the routes for this job
    for job_id in args.job_id:
        logging.info('Processing job {}'.format(job_id))
        if args.outdir is not None:
            outdir = os.path.join(args.outdir, str(job_id))
        else:
            outdir = os.path.join('.', 'tmp', str(job_id))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logging.info('Saving report in {}'.format(outdir))

        routes, routes_md = get_routes_and_metadata_from_job(connection, job_id)
        write_job_report(list(zip(routes, routes_md)), outdir)
        # TODO: Modify
        plot_routes(routes, os.path.join(outdir, "full_graph"))
