# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:09:19 2025

@author: krodberg
"""

year = 2010
pixel_tuple = (10060638, 10061112, 10061586, 10062060, 10062534, 10063008, 
               10063482, 10063956, 10064430, 10064904, 10065378, 10065852)

sqlQ = f'''WITH 
               selected_pixels AS (
                   SELECT pixel_id, 
                          pixel_centroid_x AS x, pixel_centroid_y AS y
                     FROM nrd_pixel
                    WHERE pixel_id 
                       IN {pixel_tuple}
               ),
               date_series AS (
                   SELECT TO_CHAR(DATE '{year}-01-01'+LEVEL-1, 
                                  'yyyy-mm-dd') AS da
                     FROM dual
                  CONNECT BY DATE '{year}-01-01'+LEVEL-1 < 
                               DATE '{year + 1}-01-01'
               ),
               base_grid AS (
                   SELECT p.pixel_id, p.x, p.y, d.da
                     FROM selected_pixels p
                    CROSS JOIN date_series d
               ),
               ts_agg AS (
                   SELECT nts.featureid, TRUNC(nts.tsdatetime) AS ts_date,
                          SUM(nts.tsvalue) AS sumtsvalue
                     FROM nrd_time_series nts
                    WHERE nts.featureid 
                       IN {pixel_tuple}
                      AND nts.tstypeid = 3
                      AND nts.tsdatetime >= DATE '{year}-01-01'
                      AND nts.tsdatetime < DATE '{year + 1}-01-01'
                    GROUP BY nts.featureid, TRUNC(nts.tsdatetime)
               )
           SELECT bg.pixel_id, bg.x, bg.y, bg.da,
           /* COALESCE(..,..) converts NULLS from LEFT JOIN to 0 */
                 COALESCE(ta.sumtsvalue, 0) AS value
             FROM base_grid bg
             LEFT JOIN ts_agg ta
               ON bg.pixel_id = ta.featureid
              AND bg.da = TO_CHAR(ta.ts_date, 'yyyy-mm-dd')
            ORDER BY bg.pixel_id, bg.da;
            '''