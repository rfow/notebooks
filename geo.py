import pandas as pd
import requests

API_KEY = 'AIzaSyBanal4ryTOE54WfIv5a2IEqha_lipLDXY'
input_filename = 'C:\\Users\\RiMi\\PycharmProjects\\jupyter_test\\data\\tailings_failures.csv'
output_filename = 'C:\\Users\\RiMi\\Desktop\\test.csv'
address_column_name = 'facility_loc'
data = pd.read_csv(input_filename, encoding='utf8')
addresses = data[address_column_name].tolist()


def get_google_results(address, api_key):
    geocode_url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}'.format(address)
    geocode_url = geocode_url + '&key={}'.format(api_key)
    results = requests.get(geocode_url)
    results = results.json()

    # if there's no results or an error, return empty results.
    if len(results['results']) == 0:
        output = {
            "formatted_address": None,
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "google_place_id": None,
            "type": None,
        }
    else:
        answer = results['results'][0]
        output = {
            "formatted_address": answer.get('formatted_address'),
            "latitude": answer.get('geometry').get('location').get('lat'),
            "longitude": answer.get('geometry').get('location').get('lng'),
            "accuracy": answer.get('geometry').get('location_type'),
            "google_place_id": answer.get("place_id"),
            "type": ",".join(answer.get('types')),
        }

    # Append some other details:
    output['input_string'] = address
    output['number_of_results'] = len(results['results'])
    output['status'] = results.get('status')

    return output


if __name__ == '__main__':
    # check api
    test_result = get_google_results("London, England", API_KEY)
    if (test_result['status'] != 'OK') or (test_result['formatted_address'] != 'London, UK'):
        raise ConnectionError('Problem with test results from Google Geocode - check your API key and internet connection.')

    results = []
    for address in addresses:
            try:
                geocode_result = get_google_results(address, API_KEY)
                results.append(geocode_result)
            except:
                results.append('No information')
    pd.DataFrame(results).to_csv(output_filename, encoding='utf8')
