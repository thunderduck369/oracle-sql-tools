SELECT 
    admin.APP_NO,
    wu_alloc.REQ_TOTAL_ANL_ALLOC annual_alloc,
    wu_alloc.REQ_TOTAL_MAX_MON_ALLOC max_mon_alloc,
    wu_alloc.REQ_TOTAL_MAX_DAY_ALLOC max_day_alloc
FROM REG.admin_table admin
    INNER JOIN REG.APP_COUNTIES cnty ON admin.APP_NO = cnty.ADMIN_APP_NO
    INNER JOIN REG.APP_LANDUSES APP_LANDUSES ON admin.APP_NO = APP_LANDUSES.ADMIN_APP_NO
    INNER JOIN WU_ANNUAL_ALLOCATION wu_alloc ON wu_alloc.ADMIN_APP_NO = admin.APP_NO
WHERE cnty.PRIORITY = 1
    AND APP_LANDUSES.USE_PRIORITY = 1
    AND admin.APP_STATUS = 'COMPLETE'
    AND APP_LANDUSES.LU_CODE IN ('PWS','NUR','LIV','AQU','AGR','LAN','GOL','REC','IND','COM',
                                  'PPG','PPO','PPR','PPM','DIV','DI2')
    AND cnty.CNTY_CODE IN {cnty_code}