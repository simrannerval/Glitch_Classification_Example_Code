#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH -t 4:00:00
#SBATCH --qos tiger-short

#change to your scratch directory
#SBATCH --output=/scratch/gpfs/sn0543/stellar_flare_sims/for_paper/final_sims/maps_3000_%j.out
#SBATCH --mail-user=simran.nerval@mail.utoronto.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

set -e

tod_num="1566115984.1566125361"
ar="ar5"

freq="f150"

tod="${tod_num}.${ar}:${freq}"

timecode="15661"

amp="3000"

#just for plotting cutouts because we know where the injected source it
ra=79.02308333
dec=-29.36022

#change below directories as needed
basedir="/home/sn0543/stellar_flare_sims"

datadir="${basedir}/TOD_files_for_stats/amp_${amp}/"

outdir="${basedir}/for_paper/final_sims/amp_${amp}/"

#where the trained forest is located
forestdir="${basedir}"

#where the simulated TODs are located
toddir="/scratch/gpfs/eh9397/stellar_flare_sim/final_sims/"

#where the mapmaking info is located
mapdatadir="/home/snaess/project/actpol/mapdata"

#where the classification code is located
classcodedir="/home/sn0543/Glitch_Classification/classification_code"

#where the cuts and mapmaking code is located
mapcodedir="${basedir}"

mkdir -p "${datadir}"
mkdir -p "${outdir}"
mkdir -p "${outdir}/depth_1_cutouts"


halflife="100"

echo "h = ${halflife}"

echo "${tod}" > ${datadir}/sims_${tod_num}_${ar}_${freq}_amp${amp}_h${halflife}_TOD.txt

module purge
module use --append /home/mfh2/shared/modulefiles
module use --append /projects/ACT/yilung/modulefiles

module load cutslib
module load enki_yilun
module load anaconda3/2020.7
module load so_stack/210707

export DOT_MOBY2=$HOME/.moby2

cd "${classcodedir}"


echo "${tod}" > ${datadir}/sims_${tod_num}_${ar}_${freq}_amp${amp}_h${halflife}_TOD.txt

cd "${classcodedir}"

python sims_compute_filtering_values.py --datadir "${datadir}"  --outputdir "${outdir}" --todfile "sims_${tod_num}_${ar}_${freq}_amp${amp}_h${halflife}_TOD.txt" --tod_sim_file "${toddir}" --output_df_name "df_sims_${tod}_amp${amp}_h${halflife}" --half_life "${halflife}" --amp "${amp}" --ACT

echo "Computed stats"


python Forest_and_classify.py --datadir "${outdir}"  --outputdir "${outdir}" --forestdir "${forestdir}" --trained_forest "forest_trained_ntrees50_maxdepth_15_numdet_extentratio_yadj_y0.1_meanabscorr_meanabstimelag_numberofpeaks_oct092024_df_train_afterAL_numdet_extentratio_yadj_y0.1_meanabscorr_meanabstimelag_numpeaks_oct092024.pkl" --df_classify "df_sims_${tod}_amp${amp}_h${halflife}_naffected_4.csv" --trained

echo "Classified glitches"


mkdir -p "${outdir}/new_cuts_depth1/h${halflife}/${timecode}"

mkdir -p "${outdir}/new_cuts_depth1_notmodified/h${halflife}/${timecode}"

cd "${mapcodedir}"

mpirun -n 1 python making_cuts_objects.py --datadir "${outdir}"  --outputdir "${outdir}" --toddir "${toddir}" --tod "${tod}" --time_code "${timecode}" --half_life "${halflife}" --amp "${amp}"

echo "Made new cuts objects"


cat << EOF >> ${outdir}/override_h${halflife}_notmodified.txt
cut_basic = cut_quality = cut_noiseest = "${outdir}/new_cuts_depth1_notmodified/h${halflife}/{t5}/{id}_{freq}.cuts"
cut           = {"type":"union", "subs":[cut_quality, "{skn}/sidelobe_cut/sidelobe_cut_dr6v4_pa{pa}_{freq}_20230316.hdf:cuts"]}
EOF

cat << EOF >> ${outdir}/override_h${halflife}.txt
cut_basic = cut_quality = cut_noiseest = "${outdir}/new_cuts_depth1/h${halflife}/{t5}/{id}_{freq}.cuts"
cut           = {"type":"union", "subs":[cut_quality, "{skn}/sidelobe_cut/sidelobe_cut_dr6v4_pa{pa}_{freq}_20230316.hdf:cuts"]}
EOF

mpirun -n 1 python depth_1_for_sims.py @${datadir}/sims_${tod_num}_${ar}_${freq}_amp${amp}_h${halflife}_TOD.txt --file_override=@${outdir}/override_h${halflife}_notmodified.txt ${mapdatadir}/area/advact.fits ${outdir}/sims_h${halflife} --verbosity=2 --niter=10 --tod_sim_file "${toddir}/${tod}_amp${amp}_h${halflife}.npy"

echo "Made regular maps"


mpirun -n 1 python depth_1_for_sims.py @${datadir}/sims_${tod_num}_${ar}_${freq}_amp${amp}_h${halflife}_TOD.txt --file_override=@${outdir}/override_h${halflife}.txt ${mapdatadir}/area/advact.fits ${outdir}/sims_h${halflife}_modifiedcuts --verbosity=2 --niter=10 --tod_sim_file "${toddir}/${tod}_amp${amp}_h${halflife}.npy"

echo "Made maps with modified cuts"


depth1_map=$(ls -ltr ${outdir}/sims_h${halflife}/${timecode}/*map.fits | tail -n 1)

depth1_string="${depth1_map}"

depth1_name=${depth1_string: -35: -9}

mpirun -n 1 python depth_1_maps_boxes.py --datadir ${outdir} --outputdir "${outdir}/depth_1_cutouts/" --time_code "${timecode}" --half_life "${halflife}" --ra ${ra} --dec ${dec} --depth_1 "${depth1_name}" --amp "${amp}"

echo "Made depth-1 cutouts" 