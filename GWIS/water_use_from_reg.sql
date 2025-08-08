SELECT admin.app_no,
   sub.APPLIES_TO_DATE,
   sub.SUBM_VALUE data_value,
   sub.TMU_CODE data_value_units,
   req.TRE_ID subm_type,
   tlreq.NAME req_name, 
   req.WALR_TYPE report_type,
   admin.permit_no site_id
FROM REG.admin admin
   INNER JOIN REG.WU_APP_FACILITY apfac ON apfac.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.APP_COUNTIES cnty ON cnty.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.APP_LC_DEFS lc ON lc.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.APP_LANDUSES lu ON lu.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.WU_FAC_INV fac ON fac.ID = apfac.FACINV_ID
   INNER JOIN REG.WUC_APP_LC_REQMTS req ON req.APPLC_ADMIN_APP_NO = admin.APP_no
   INNER JOIN REG.tl_counties tl_counties ON tl_counties.COUNTY_CODE = cnty.CNTY_CODE
   INNER JOIN REG.tl_sources src ON src.ID = apfac.SOURCE_ID
   INNER JOIN REG.WUC_APP_SIM_SUBMS sub ON sub.WALR_REQM_ID = req.REQM_ID
   INNER JOIN REG.TL_REQUIREMENTS tlreq ON tlreq.ID = req.TL_LC_REQM_ID
WHERE lc.ADMIN_APP_NO = req.APPLC_ADMIN_APP_NO
   AND to_char(apFac.FACINV_ID )= req.REQM_ENT_KEY1
   AND req.TRE_ID IN (3, 4)
   AND lc.ID = req.APPLC_ID   
   AND sub.subm_value IS NOT NULL
   AND cnty.PRIORITY LIKE 1
   AND lu.USE_PRIORITY LIKE 1     
   AND sub.APPLIES_TO_DATE 
   BETWEEN TO_DATE('01/01/1985','MM/DD/YYYY') 
       AND TO_DATE('12/31/2024','MM/DD/YYYY')
   AND (tlreq.NAME LIKE 'Water Use Report%')
   AND lu.LU_CODE IN ('PWS','NUR','LIV','AQU','AGR','LAN','GOL','REC','IND','COM',
                              'PPG','PPO','PPR','PPM','DIV','DI2')
   AND cnty.CNTY_CODE IN {cnty_code}
   AND req.WALR_TYPE LIKE 'SIMPLE'
   AND admin.APP_STATUS = 'COMPLETE'
UNION
SELECT admin.app_no,
   sub.APPLIES_TO_DATE,
   sub.SUBM_VALUE data_value,
   sub.TMU_CODE data_value_units,
   req.TRE_ID subm_type,
   tlreq.NAME req_name, 
   req.WALR_TYPE report_type,
   admin.permit_no site_id
FROM REG.admin admin
   INNER JOIN REG.WU_APP_FACILITY apfac ON apfac.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.APP_COUNTIES cnty ON cnty.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.APP_LC_DEFS lc ON lc.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.APP_LANDUSES lu ON lu.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN REG.WU_FAC_INV fac ON fac.ID = apfac.FACINV_ID
   INNER JOIN REG.WUC_APP_LC_REQMTS req ON req.APPLC_ADMIN_APP_NO = admin.APP_no
   INNER JOIN REG.tl_counties tl_counties ON tl_counties.COUNTY_CODE = cnty.CNTY_CODE
   INNER JOIN REG.tl_sources src ON src.ID = apfac.SOURCE_ID
   INNER JOIN REG.WUC_APP_SIM_SUBMS sub ON sub.WALR_REQM_ID = req.REQM_ID
   INNER JOIN REG.TL_REQUIREMENTS tlreq ON tlreq.ID = req.TL_LC_REQM_ID
   INNER JOIN WU_ANNUAL_ALLOCATION ON WU_ANNUAL_ALLOCATION.ADMIN_APP_NO = admin.APP_NO
   INNER JOIN STAFF_ANNUAL_ALLOC ON STAFF_ANNUAL_ALLOC.APP_NO = admin.APP_NO
WHERE lc.ADMIN_APP_NO = req.APPLC_ADMIN_APP_NO
   AND req.TRE_ID = 1
   AND lc.ID = req.APPLC_ID   
   AND cnty.PRIORITY LIKE 1
   AND lu.USE_PRIORITY LIKE 1     
   AND sub.APPLIES_TO_DATE between to_date('01/01/1985','MM/DD/YYYY') AND to_date('12/31/2024','MM/DD/YYYY')
   AND (tlreq.NAME LIKE 'Water Use Report%')
   AND lu.LU_CODE IN ('PWS','NUR','LIV','AQU','AGR','LAN','GOL','REC','IND','COM',
                              'PPG','PPO','PPR','PPM','DIV','DI2')
   AND req.WALR_TYPE LIKE 'TOTAL'
   AND sub.TMU_CODE IN ('MG/MONTH')
   AND sub.SUBM_VALUE IS NOT NULL
   AND cnty.CNTY_CODE in {cnty_code}
   AND admin.APP_STATUS = 'COMPLETE'
   