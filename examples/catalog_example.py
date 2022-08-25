#!/usr/bin/env python



from wildhunt import catalog_new






if __name__ == '__main__':



    filename = 'UKIDSS_sources.csv'
    ra_colname = 'RA'
    dec_colname = 'DEC'


    cat = catalog_new.Catalog('example', ra_column_name=ra_colname,
                              dec_column_name=dec_colname,
                              id_column_name='Name',
                              filename=filename,
                              chunksize=50)



    cat.cross_match(survey='DELS')


