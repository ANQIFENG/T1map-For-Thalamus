#!/bin/bash
mprage_path=../mtbi/MTBI-MRCON0001/v1/nii/MTBI-MRCON0001_v1_01-01_BRAIN-T1-IRFSPGR-3D-SAGITTAL-PRE.nii.gz
fgatir_path=../mtbi/MTBI-MRCON0001/v1/nii/MTBI-MRCON0001_v1_02-01_BRAIN-FGATIR-IRFSPGR-3D-SAGITTAL-PRE.nii.gz
out_dir=../results
sif=smri_pipeline_v0.0.19.sif # Built by the Dockerfile and run-pipeline

start_time=$(date +%s)
singularity run -e --nv -B /iacl $sif \
            --mprage ${mprage_path} \
            --fgatir ${fgatir_path} \
            --out_dir ${out_dir}\
            --tr 4000.0\
            --ti_mprage 1400.0\
            --ti_fgatir 400.0

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Execution time: $duration seconds"