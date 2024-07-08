#!/usr/bin/env python

import time

import requests
import getpass

def esaotf_login():
    from http.cookiejar import MozillaCookieJar
    import requests

    cookies = MozillaCookieJar()

    data = {
        'username': 'username',
        'password': 'password',
    }

    with requests.Session() as session:
        session.cookies = cookies
        response = session.post('https://easotf.esac.esa.int/sas-dd/login', data=data, verify=False)
        cookies.save('cookies.txt', ignore_discard=True, ignore_expires=True)


if __name__ == '__main__':

    ra = 234.790709946306 # Right ascension in decimal degrees
    dec = 73.7210875690683 # Declination in decimal degrees

    username = input('Enter user name (+ENTER): ')
    pwd = getpass.getpass('Enter password (+ENTER): ')

    from http.cookiejar import MozillaCookieJar


    adql_query = """SELECT observation_stack.file_name, observation_stack.observation_stack_oid,
                    observation_stack.observation_id, observation_stack.ra, observation_stack.dec,
                    observation_stack.instrument_name, observation_stack.filter_name, observation_stack.release_name,
                    observation_stack.category, observation_stack.second_type, observation_stack.technique,
                    observation_stack.product_type, observation_stack.start_time, observation_stack.duration
                    FROM sedm.observation_stack WHERE (product_type like '%Stacked%') 
                    AND ((observation_stack.fov IS NOT NULL 
                    AND INTERSECTS(CIRCLE('ICRS',232.1000787837041,71.99615882692106,0.08333333333333333),observation_stack.fov)=1))
                    ORDER BY observation_id ASC"""



    cookies = MozillaCookieJar()

    data = {
        'username': username,
        'password': pwd,
    }

    # with requests.Session() as session:
    #     session.cookies = cookies
    #     response = session.post('https://easotf.esac.esa.int/sas-dd/login', data=data, verify=False)
    #     cookies.save('cookies.txt', ignore_discard=True, ignore_expires=True)

    with requests.Session() as session:
        session.cookies = cookies
        response = session.post('https://easotf.esac.esa.int/tap-server/login', data=data, verify=False)
        cookies.save('cookies.txt', ignore_discard=True, ignore_expires=True)

    response = requests.get('https://easotf.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv'
                            '&QUERY={}'.format(adql_query.replace(' ', '+')), cookies=cookies, verify=False)

    # from IPython import embed
    # embed()

    with requests.Session() as session:
        session.cookies = cookies
        response = session.post('https://easotf.esac.esa.int/sas-cutout/login', data=data, verify=False)
        cookies.save('cookies.txt', ignore_discard=True, ignore_expires=True)

        print(response.text)

    response = requests.get('https://easotf.esac.esa.int/sas-sia/sia2/query?POS=BOX,187.89,29.54,20.0')



    print(response.text)

    response = requests.get(
        'https://easotf.esac.esa.int/sas-cutout/cutout?filepath=/data/repository/NIR/19704/EUC_NIR_W-STACK_NIR-J'
        '-19704_20190718T001858.5Z_00.00.fits&collection=NISP&obsid=19704&POS=CIRCLE,187.89,29.54,0.0013888888',
        verify=False,
        cookies=cookies,
    )

    # response = requests.get(
    #     'https://easotf.esac.esa.int/sas-cutout/cutout?filepath=/data/repository/NIR/1299/EUC_NIR_W-STK-IMAGE_J'
    #     '-1299_20240503T204942.296546Z.fits&collection=NISP&obsid=1299&POS=CIRCLE,232.6337194201,72.26425317565,0.0833',
    #     verify=False,
    #     cookies=cookies,
    # )

    print('image download response', response.status_code)
    print(response.content)

    if response.status_code == 200:
        with open('test2.fits', 'wb') as f:
            f.write(response.content)





