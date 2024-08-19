import clip
import os
import numpy as np
from io import BytesIO
import base64
import asyncio
import gzip

c = clip.Clip( False, None )

with open( "testdata/metadata.csv", "rb" ) as metadata, open( "testdata/train/train_1/train_1_a/train_1_a_1/train_1_a_1.nii.gz", "rb" ) as nifti:

    nifti_str = base64.b64encode( gzip.open( nifti ).read() )
    metadata_str = base64.b64encode( metadata.read() )

    inp = clip.ClipInput()
    inp.texts = [ "" ]
    inp.images = [ [ "nifti", nifti_str, metadata_str ] ]

    out = asyncio.run( c.vectorize( inp ) )
    print( out )


inp = clip.ClipInput()
inp.texts = [ "" ]
inp.images = [ [ "dicom", [] ] ]

for i in os.listdir( "testdata/dicom" ):
    with open( os.path.join( "testdata", "dicom", i ), "rb" ) as dcm:
        dcm_str = base64.b64encode( dcm.read() )
        inp.images[ 0 ][ 1 ].append( dcm_str )

out = asyncio.run( c.vectorize( inp ) )
print( out )
