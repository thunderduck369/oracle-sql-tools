WITH RankedApps AS (
  SELECT
     admin.app_no,
     TO_CHAR (apfac.FACINV_ID, '9999999') FACINV_ID,
     fac.FACINV_TYPE,
     fac.NAME fac_Name, 
     fac.facwlsts_code facility_status,
     fac.CASED_DEPTH,
     fac.WELL_DEPTH,
     fac.PUMP_COORDX,
     fac.PUMP_COORDY,
     fac.PUMP_INTAKE_DEPTH,
     fac.TOP_OF_CASING,
     fac.MEAS_PT_ELEV,
     src.id source_id,
     admin.FINAL_ACTION_DATE,
     ROW_NUMBER() OVER (
       PARTITION BY apfac.FACINV_ID
       ORDER BY 
         CASE WHEN admin.FINAL_ACTION_DATE IS NULL THEN 1 ELSE 0 END,
         admin.FINAL_ACTION_DATE DESC
     ) AS rn
  FROM REG.admin admin
     INNER JOIN REG.WU_APP_FACILITY apfac ON apfac.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.APP_COUNTIES cnty ON cnty.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.APP_LC_DEFS lc ON lc.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.APP_LANDUSES lu ON lu.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.WU_FAC_INV fac ON fac.ID = apfac.FACINV_ID
     INNER JOIN REG.WUC_APP_LC_REQMTS req ON req.APPLC_ADMIN_APP_NO = admin.APP_no
     INNER JOIN REG.tl_counties tl_counties ON tl_counties.COUNTY_CODE = cnty.CNTY_CODE
     INNER JOIN REG.tl_sources src ON src.ID = apfac.SOURCE_ID
     INNER JOIN REG.TL_REQUIREMENTS tlreq ON tlreq.ID = req.TL_LC_REQM_ID
     INNER JOIN REG.WU_FAC_STS_TRK sts ON sts.facinv_id = apfac.facinv_id
  WHERE lc.ADMIN_APP_NO = req.APPLC_ADMIN_APP_NO
     AND TO_CHAR(apFac.FACINV_ID) = req.REQM_ENT_KEY1
     AND lc.ID = req.APPLC_ID   
     AND cnty.PRIORITY LIKE 1
     AND lu.USE_PRIORITY LIKE 1
     AND tlreq.NAME LIKE 'Water Use Report%'
     AND admin.APP_STATUS = 'COMPLETE'
     AND lu.LU_CODE IN (
        'PWS','NUR','LIV','AQU','AGR','LAN','GOL','REC','IND','COM',
        'PPG','PPO','PPR','PPM','DIV','DI2'
     )
     AND cnty.CNTY_CODE IN (6, 8, 11, 13, 22, 26, 28, 36, 43, 44, 47, 48, 49, 50, 53, 56)
)
SELECT app_no, facinv_id, facinv_type, fac_name, facility_status, cased_depth, well_depth, pump_coordx, pump_coordy, pump_intake_depth, top_of_casing, meas_pt_elev, source_id
FROM RankedApps
WHERE rn = 1
