import mysql.connector
import os
import yaml
import pandas as pd
import datetime

database_cfg = yaml.full_load(open(os.path.join(os.getcwd(), "database_config.yml"), 'r'))


def get_clips_from_db(table_name, sliding=True):
    '''
    Pulls clips from database using specified MySQL query.
    :param table_name: Str, indicates location from the multi-center study for which to pull clips.
    :param sliding: Bool, set to True to pull present lung sliding clips. Set to False to pull absent lung sliding clips.
    '''
    # Get database configs
    user = database_cfg['USERNAME']
    password = database_cfg['PASSWORD']
    host = database_cfg['HOST']
    db = database_cfg['DATABASE']

    # Establish connection to database
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=db)

    # initialize pleural_line_findings query condition
    pleural_line_findings = " is null OR pleural_line_findings='thickened'"

    # adjust pleural_line_findings query condition as needed
    if table_name == 'clips_chile':
        if sliding:
            pleural_line_findings = "='' OR pleural_line_findings='thickened'"
    if not sliding:
        pleural_line_findings = "='absent_lung_sliding'"

    # set up the MySQL query
    query = "SELECT * FROM {} WHERE view='parenchymal' AND a_or_b_lines='a_lines'" \
            " AND (quality NOT LIKE '%significant_probe_movement%' OR quality is null)" \
            " AND (pleural_line_findings{}"\
        .format(table_name, pleural_line_findings)

    # If pulling Alberta data, ensure that these clips have been reviewed and are ready to use!
    if table_name == 'clips_alberta':
        query += ' AND reviewed=1 AND do_not_use=0'
        if not sliding:
            pleural_line_findings += "='absent_lung_sliding' OR pleural_line_findings='thickened|absent_lung_sliding'"

    # Complete query syntax
    query += ');'

    # pull clips from the database into a DataFrame
    clips_df = pd.read_sql(query, cnx)
    return clips_df


def add_date_to_filename(file):
    '''
    Labels a file name with useful information about date and time.
    :param file: name of file as string
    :return labelled_filename: name of file joined with date and time info.
    '''
    cur_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    labelled_filename = file + '_' + cur_date
    return labelled_filename
