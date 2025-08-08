Select
         app_no,
         field_id,
         irr_eff,
         LU_CODE as landuse_code,
         field_acres,
         crop_name,
         use_priority,
         irr_type
         from
  (SELECT distinct app_counties.cnty_code,
         tl_counties.county,
         ADMIN.permit_no,
         ADMIN.app_no,
         app_fees.eff_date,
         ADMIN.FINAL_ACTION_DATE,
         ADMIN.EXPIRATION_DATE,
         ADMIN.project_name,
         wu_app_fields.ID field_id,
         wu_alloc.rec_total_anl_alloc ANN_ALLOC,
         WU_ALLOC.REC_TOTAL_AVG_DAY_ALLOC AVG_DAY_ALLOC,
         wu_alloc.rec_total_max_day_alloc MAX_DAY_ALLOC,
         ADMIN.PROJECT_ACREAGE,
         ADMIN.acres_served total_acres_served,
         wu_app_fields.acres field_acres,
         tl_crops.NAME crop_name,
         TL_IRR_SYS.NAME irr_type,
         wu_app_fields.irr_eff,
         APP_LANDUSES.LU_CODE,
         wu_app_fields.applu_lu_code,
         use_priority
    FROM reg.ADMIN ADMIN,
         reg.app_counties app_counties, reg.APP_LANDUSES APP_LANDUSES,
         (  SELECT adm.permit_no, MAX (adm.app_no) app_no
              FROM reg.ADMIN ADM
             WHERE     adm.app_status = 'COMPLETE'
                   AND ADM.EXPIRATION_DATE >
                          TO_DATE ('12/31/2017', 'MM/DD/YYYY')
                   AND adm.final_action_date <=
                          TO_DATE ('12/31/2019', 'MM/DD/YYYY')
          GROUP BY adm.permit_no) lastApp,
         reg.tl_counties tl_counties,
         reg.app_fees app_fees,
         reg.tl_fee_codes tl_fee_codes,
         reg.wu_annual_allocation wu_alloc,
         reg.tl_crops,
         reg.wu_app_fields wu_app_fields,
         reg.tl_irr_sys tl_irr_sys,
         reg.wu_app_crops wu_app_crops,
         reg.wu_app_plantings wu_app_plant
   WHERE     admin.app_no = lastApp.app_no
         AND ADMIN.app_no = app_fees.admin_app_no
         AND TL_IRR_SYS.CODE = WU_APP_FIELDS.TLIRRSYS_CODE
         AND app_fees.fee_code = tl_fee_codes.fee_code
         AND ADMIN.app_no = wu_alloc.admin_app_no
         AND ADMIN.app_no = app_counties.admin_app_no
         AND wu_app_crops.appflds_app_no = wu_app_fields.admin_app_no
         AND WU_APP_PLANT.APPCROP_APP_NO = wu_app_crops.appflds_app_no
         AND WU_APP_PLANT.APPCROP_ID = WU_APP_CROPS.APPFLDS_ID
         AND wu_app_fields.admin_app_no = ADMIN.app_no
         AND wu_app_crops.appflds_id = wu_app_fields.ID
         AND wu_app_crops.tlcrop_code = tl_crops.code
         AND app_counties.cnty_code = tl_counties.county_code
         AND (    (app_fees.eff_date BETWEEN tl_fee_codes.eff_date
                                         AND NVL (tl_fee_codes.eff_to_date,
                                                  app_fees.eff_date))
              AND (tl_fee_codes.fee_permit_type = 'WU'))
         AND APP_COUNTIES.CNTY_CODE = TL_COUNTIES.COUNTY_CODE
   AND ADMIN.APP_NO = APP_LANDUSES.ADMIN_APP_NO
   AND ADMIN.APP_NO = APP_COUNTIES.ADMIN_APP_NO
   AND use_priority = 1
   AND (
   APP_LANDUSES.lu_code IN ('AGR','AQU','LIV','NUR') )
        --      APPLU_LU_CODE in ('AGR','AQU','LIV','NUR'))
         AND app_counties.priority = 1
         AND APP_COUNTIES.CNTY_CODE IN (6, 8, 11, 13, 22, 26, 28, 36, 43, 44, 47, 48, 49, 50, 53, 56)      
)