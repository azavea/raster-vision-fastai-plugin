import boto3
import requests

from rastervision.utils.files import list_paths

def get_session(refresh_token):
    """Helper method to create a requests Session"""

    post_body = {'refresh_token': refresh_token}
    response = requests.post('https://app.rasterfoundry.com/api/tokens/', json=post_body)
    response.raise_for_status()
    token = response.json()['id_token']

    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer {}'.format(token)})
    return session

# Create payload to POST to Raster Foundry API
def create_upload_from_s3_path(s3_path, datasource_id, project_id, metadata={}):
    return dict(
        uploadStatus='UPLOADED',
        files=[s3_path],
        uploadType='S3',
        fileType='GEOTIFF',
        datasource=datasource_id,
        metadata=metadata,
        visibility='PRIVATE',
        projectId=project_id
    )

if __name__ == '__main__':
    # Fill this in:
    refresh_token = ''
    potsdam_uri = 's3://raster-vision-raw-data/isprs-potsdam/4_Ortho_RGBIR/'
    # RGB
    datasource_id = '4cf6748b-e709-471e-a6cf-f50e4259b2cd'
    project_id = '10f0a913-4ef1-4222-875b-4b2b8fc5963a'

    session = get_session(refresh_token)
    image_uris = list_paths(potsdam_uri, ext='tif')
    for image_uri in image_uris:
        print('Posting upload of {}'.format(image_uri))
        upload = create_upload_from_s3_path(image_uri, datasource_id, project_id)
        response = session.post('https://app.rasterfoundry.com/api/uploads/', json=upload)
        response.raise_for_status()