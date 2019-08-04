# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_app]
from flask import Flask, url_for
from flask import request, render_template
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor, Pool
import xgboost
import pandas as pd
import joblib
import numpy as np
import os

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

@app.route('/', methods=['GET', 'POST'])
def predictor():
    try:
        if request.method == 'POST':
            light = lgb.Booster(model_file='light_model.txt')
            xgb = xgboost.Booster(model_file='xgb_model.txt')
            light_meta_model = lgb.Booster(model_file='lgb_meta_model.txt')
            svr_meta_model = joblib.load('svr_meta_model.joblib')

            df = {}

            p_zone = request.form['p_zone']
            d_zone = request.form['d_zone']
            p_day_of_month = request.form['p_day_of_month']
            p_hour = request.form['p_hour']
            vendorid = request.form['vendorid']
            passenger_count = request.form['passenger_count']
            day_avg_temperature = request.form['avg_temp']
            day_precipitation = request.form['precipitation']
            day_new_snow = request.form['new_snow']
            snow_depth = request.form['snow_depth']

            p_zone_id = p_zone.split(':')[0].strip()
            d_zone_id = d_zone.split(':')[0].strip()

            if str(p_day_of_month) == "1":
                p_day_name = "Thursday"
            elif str(p_day_of_month) == "4":
                p_day_name = "Sunday"
            else:
                p_day_name = "Monday"

            zones_lookup_table = pd.read_csv('taxi_zone_lookup.csv')
            zones_lookup_table = zones_lookup_table[~zones_lookup_table.service_zone.isna()]

            p_borough = zones_lookup_table[zones_lookup_table.LocationID
                                           == int(p_zone_id)].Borough.squeeze()
            d_borough = zones_lookup_table[zones_lookup_table.LocationID
                                           == int(d_zone_id)].Borough.squeeze()

            p_service_zone = zones_lookup_table[zones_lookup_table.LocationID
                                                == int(p_zone_id)].service_zone.squeeze()
            d_service_zone = zones_lookup_table[zones_lookup_table.LocationID
                                                == int(d_zone_id)].service_zone.squeeze()

            df['passenger_count'] = [int(passenger_count)]

            if p_day_name in ('Saturday', 'Sunday'):
                df['pickup__is_weekend'] = [1]
            else:
                df['pickup__is_weekend'] = [0]

            df['day_temperature_range'] = 9.31 # mean from training data

            if int(passenger_count) > 6:
                df['is_passenger_count_gt_6'] = [1]
            else:
                df['is_passenger_count_gt_6'] = [0]

            if int(vendorid) == 1:
                df['VendorID__1'] = [1]
                df['VendorID__2'] = [0]
            else:
                df['VendorID__1'] = [0]
                df['VendorID__2'] = [1]

            for dn in ('Wednesday', 'Saturday', 'Sunday', 'Monday', 'Thursday'):
                if dn == p_day_name:
                    df['pickup_day_name__' + dn] = 1
                else:
                    df['pickup_day_name__' + dn] = 0

            for br in ('Bronx', 'Brooklyn', 'EWR', 'Manhattan', 'Queens', 'Staten Island'):
                if br == p_borough:
                    df['pickup_borough__' + br] = 1
                else:
                    df['pickup_borough__' + br] = 0

            for br in ('Bronx', 'Brooklyn', 'EWR', 'Manhattan', 'Queens', 'Staten Island'):
                if br == d_borough:
                    df['dropoff_borough__' + br] = 1
                else:
                    df['dropoff_borough__' + br] = 0

            for sz in ('Airports', 'Boro Zone', 'EWR', 'Yellow Zone'):
                if sz == p_service_zone:
                    df['pickup_service_zone__' + sz] = 1
                else:
                    df['pickup_service_zone__' + sz] = 0

            for sz in zones_lookup_table.service_zone.unique():
                if sz == d_service_zone:
                    df['dropoff_service_zone__' + sz] = 1
                else:
                    df['dropoff_service_zone__' + sz] = 0

            for per_s, per_e in [(12,14), (15,17), (18,20), (21,23)]:
                if per_s <= int(p_hour) <= per_e:
                    df['pickup_period_of_day__{}_{}'.format(per_s, per_e)] = 1
                    # for target encoding
                    df['pickup_period_of_day'] = str(per_s) + '_' + str(per_e)
                else:
                    df['pickup_period_of_day__{}_{}'.format(per_s, per_e)] = 0

            for dom in (1,4,5):
                if int(p_day_of_month) == dom:
                    df['pickup_day_of_month__{}'.format(dom)] = 1
                else:
                    df['pickup_day_of_month__{}'.format(dom)] = 0

            PULocIDs = ['PULocationID__1', 'PULocationID__3', 'PULocationID__4', 'PULocationID__6', 'PULocationID__7', 'PULocationID__8', 'PULocationID__9', 'PULocationID__10', 'PULocationID__12', 'PULocationID__13', 'PULocationID__14', 'PULocationID__17', 'PULocationID__18', 'PULocationID__19', 'PULocationID__20', 'PULocationID__21', 'PULocationID__22', 'PULocationID__24', 'PULocationID__25', 'PULocationID__26', 'PULocationID__28', 'PULocationID__29', 'PULocationID__31', 'PULocationID__32', 'PULocationID__33', 'PULocationID__34', 'PULocationID__35', 'PULocationID__36', 'PULocationID__37', 'PULocationID__38', 'PULocationID__39', 'PULocationID__40', 'PULocationID__41', 'PULocationID__42', 'PULocationID__43', 'PULocationID__45', 'PULocationID__47', 'PULocationID__48', 'PULocationID__49', 'PULocationID__50', 'PULocationID__51', 'PULocationID__52', 'PULocationID__53', 'PULocationID__54', 'PULocationID__55', 'PULocationID__56', 'PULocationID__57', 'PULocationID__58', 'PULocationID__60', 'PULocationID__61', 'PULocationID__62', 'PULocationID__63', 'PULocationID__65', 'PULocationID__66', 'PULocationID__67', 'PULocationID__68', 'PULocationID__69', 'PULocationID__70', 'PULocationID__71', 'PULocationID__72', 'PULocationID__73', 'PULocationID__74', 'PULocationID__75', 'PULocationID__76', 'PULocationID__77', 'PULocationID__78', 'PULocationID__79', 'PULocationID__80', 'PULocationID__81', 'PULocationID__82', 'PULocationID__83', 'PULocationID__85', 'PULocationID__87', 'PULocationID__88', 'PULocationID__89', 'PULocationID__90', 'PULocationID__91', 'PULocationID__92', 'PULocationID__93', 'PULocationID__94', 'PULocationID__95', 'PULocationID__96', 'PULocationID__97', 'PULocationID__98', 'PULocationID__100', 'PULocationID__101', 'PULocationID__102', 'PULocationID__106', 'PULocationID__107', 'PULocationID__108', 'PULocationID__112', 'PULocationID__113', 'PULocationID__114', 'PULocationID__116', 'PULocationID__119', 'PULocationID__120', 'PULocationID__121', 'PULocationID__122', 'PULocationID__123', 'PULocationID__124', 'PULocationID__125', 'PULocationID__126', 'PULocationID__127', 'PULocationID__128', 'PULocationID__129', 'PULocationID__130', 'PULocationID__131', 'PULocationID__132', 'PULocationID__133', 'PULocationID__134', 'PULocationID__135', 'PULocationID__136', 'PULocationID__137', 'PULocationID__138', 'PULocationID__140', 'PULocationID__141', 'PULocationID__142', 'PULocationID__143', 'PULocationID__144', 'PULocationID__145', 'PULocationID__146', 'PULocationID__147', 'PULocationID__148', 'PULocationID__149', 'PULocationID__151', 'PULocationID__152', 'PULocationID__153', 'PULocationID__155', 'PULocationID__157', 'PULocationID__158', 'PULocationID__159', 'PULocationID__160', 'PULocationID__161', 'PULocationID__162', 'PULocationID__163', 'PULocationID__164', 'PULocationID__165', 'PULocationID__166', 'PULocationID__167', 'PULocationID__168', 'PULocationID__169', 'PULocationID__170', 'PULocationID__171', 'PULocationID__172', 'PULocationID__173', 'PULocationID__174', 'PULocationID__175', 'PULocationID__177', 'PULocationID__178', 'PULocationID__179', 'PULocationID__180', 'PULocationID__181', 'PULocationID__182', 'PULocationID__185', 'PULocationID__186', 'PULocationID__188', 'PULocationID__189', 'PULocationID__190', 'PULocationID__191', 'PULocationID__192', 'PULocationID__193', 'PULocationID__194', 'PULocationID__195', 'PULocationID__196', 'PULocationID__197', 'PULocationID__198', 'PULocationID__200', 'PULocationID__201', 'PULocationID__202', 'PULocationID__203', 'PULocationID__205', 'PULocationID__207', 'PULocationID__208', 'PULocationID__209', 'PULocationID__210', 'PULocationID__211', 'PULocationID__212', 'PULocationID__213', 'PULocationID__215', 'PULocationID__216', 'PULocationID__217', 'PULocationID__218', 'PULocationID__219', 'PULocationID__220', 'PULocationID__223', 'PULocationID__224', 'PULocationID__225', 'PULocationID__226', 'PULocationID__227', 'PULocationID__228', 'PULocationID__229', 'PULocationID__230', 'PULocationID__231', 'PULocationID__232', 'PULocationID__233', 'PULocationID__234', 'PULocationID__235', 'PULocationID__236', 'PULocationID__237', 'PULocationID__238', 'PULocationID__239', 'PULocationID__240', 'PULocationID__241', 'PULocationID__242', 'PULocationID__243', 'PULocationID__244', 'PULocationID__246', 'PULocationID__247', 'PULocationID__248', 'PULocationID__249', 'PULocationID__250', 'PULocationID__251', 'PULocationID__252', 'PULocationID__254', 'PULocationID__255', 'PULocationID__256', 'PULocationID__257', 'PULocationID__258', 'PULocationID__259', 'PULocationID__260', 'PULocationID__261', 'PULocationID__262', 'PULocationID__263', 'PULocationID__183', 'PULocationID__222', 'PULocationID__139', 'PULocationID__46', 'PULocationID__154', 'PULocationID__117', 'PULocationID__110', 'PULocationID__15', 'PULocationID__11', 'PULocationID__105', 'PULocationID__206', 'PULocationID__16']
            DOLocIDs = ['DOLocationID__1', 'DOLocationID__3', 'DOLocationID__4', 'DOLocationID__5', 'DOLocationID__6', 'DOLocationID__7', 'DOLocationID__8', 'DOLocationID__9', 'DOLocationID__10', 'DOLocationID__11', 'DOLocationID__12', 'DOLocationID__13', 'DOLocationID__14', 'DOLocationID__15', 'DOLocationID__16', 'DOLocationID__17', 'DOLocationID__18', 'DOLocationID__19', 'DOLocationID__20', 'DOLocationID__21', 'DOLocationID__22', 'DOLocationID__23', 'DOLocationID__24', 'DOLocationID__25', 'DOLocationID__26', 'DOLocationID__27', 'DOLocationID__28', 'DOLocationID__29', 'DOLocationID__30', 'DOLocationID__31', 'DOLocationID__32', 'DOLocationID__33', 'DOLocationID__34', 'DOLocationID__35', 'DOLocationID__36', 'DOLocationID__37', 'DOLocationID__38', 'DOLocationID__39', 'DOLocationID__40', 'DOLocationID__41', 'DOLocationID__42', 'DOLocationID__43', 'DOLocationID__44', 'DOLocationID__45', 'DOLocationID__46', 'DOLocationID__47', 'DOLocationID__48', 'DOLocationID__49', 'DOLocationID__50', 'DOLocationID__51', 'DOLocationID__52', 'DOLocationID__53', 'DOLocationID__54', 'DOLocationID__55', 'DOLocationID__56', 'DOLocationID__57', 'DOLocationID__58', 'DOLocationID__59', 'DOLocationID__60', 'DOLocationID__61', 'DOLocationID__62', 'DOLocationID__63', 'DOLocationID__64', 'DOLocationID__65', 'DOLocationID__66', 'DOLocationID__67', 'DOLocationID__68', 'DOLocationID__69', 'DOLocationID__70', 'DOLocationID__71', 'DOLocationID__72', 'DOLocationID__73', 'DOLocationID__74', 'DOLocationID__75', 'DOLocationID__76', 'DOLocationID__77', 'DOLocationID__78', 'DOLocationID__79', 'DOLocationID__80', 'DOLocationID__81', 'DOLocationID__82', 'DOLocationID__83', 'DOLocationID__84', 'DOLocationID__85', 'DOLocationID__86', 'DOLocationID__87', 'DOLocationID__88', 'DOLocationID__89', 'DOLocationID__90', 'DOLocationID__91', 'DOLocationID__92', 'DOLocationID__93', 'DOLocationID__94', 'DOLocationID__95', 'DOLocationID__96', 'DOLocationID__97', 'DOLocationID__98', 'DOLocationID__100', 'DOLocationID__101', 'DOLocationID__102', 'DOLocationID__106', 'DOLocationID__107', 'DOLocationID__108', 'DOLocationID__109', 'DOLocationID__111', 'DOLocationID__112', 'DOLocationID__113', 'DOLocationID__114', 'DOLocationID__115', 'DOLocationID__116', 'DOLocationID__117', 'DOLocationID__118', 'DOLocationID__119', 'DOLocationID__120', 'DOLocationID__121', 'DOLocationID__122', 'DOLocationID__123', 'DOLocationID__124', 'DOLocationID__125', 'DOLocationID__126', 'DOLocationID__127', 'DOLocationID__128', 'DOLocationID__129', 'DOLocationID__130', 'DOLocationID__131', 'DOLocationID__132', 'DOLocationID__133', 'DOLocationID__134', 'DOLocationID__135', 'DOLocationID__136', 'DOLocationID__137', 'DOLocationID__138', 'DOLocationID__139', 'DOLocationID__140', 'DOLocationID__141', 'DOLocationID__142', 'DOLocationID__143', 'DOLocationID__144', 'DOLocationID__145', 'DOLocationID__146', 'DOLocationID__147', 'DOLocationID__148', 'DOLocationID__149', 'DOLocationID__150', 'DOLocationID__151', 'DOLocationID__152', 'DOLocationID__153', 'DOLocationID__154', 'DOLocationID__155', 'DOLocationID__156', 'DOLocationID__157', 'DOLocationID__158', 'DOLocationID__159', 'DOLocationID__160', 'DOLocationID__161', 'DOLocationID__162', 'DOLocationID__163', 'DOLocationID__164', 'DOLocationID__165', 'DOLocationID__166', 'DOLocationID__167', 'DOLocationID__168', 'DOLocationID__169', 'DOLocationID__170', 'DOLocationID__171', 'DOLocationID__172', 'DOLocationID__173', 'DOLocationID__174', 'DOLocationID__175', 'DOLocationID__176', 'DOLocationID__177', 'DOLocationID__178', 'DOLocationID__179', 'DOLocationID__180', 'DOLocationID__181', 'DOLocationID__182', 'DOLocationID__183', 'DOLocationID__184', 'DOLocationID__185', 'DOLocationID__186', 'DOLocationID__187', 'DOLocationID__188', 'DOLocationID__189', 'DOLocationID__190', 'DOLocationID__191', 'DOLocationID__192', 'DOLocationID__193', 'DOLocationID__194', 'DOLocationID__195', 'DOLocationID__196', 'DOLocationID__197', 'DOLocationID__198', 'DOLocationID__200', 'DOLocationID__201', 'DOLocationID__202', 'DOLocationID__203', 'DOLocationID__204', 'DOLocationID__205', 'DOLocationID__206', 'DOLocationID__207', 'DOLocationID__208', 'DOLocationID__209', 'DOLocationID__210', 'DOLocationID__211', 'DOLocationID__212', 'DOLocationID__213', 'DOLocationID__214', 'DOLocationID__215', 'DOLocationID__216', 'DOLocationID__217', 'DOLocationID__218', 'DOLocationID__219', 'DOLocationID__220', 'DOLocationID__221', 'DOLocationID__222', 'DOLocationID__223', 'DOLocationID__224', 'DOLocationID__225', 'DOLocationID__226', 'DOLocationID__227', 'DOLocationID__228', 'DOLocationID__229', 'DOLocationID__230', 'DOLocationID__231', 'DOLocationID__232', 'DOLocationID__233', 'DOLocationID__234', 'DOLocationID__235', 'DOLocationID__236', 'DOLocationID__237', 'DOLocationID__238', 'DOLocationID__239', 'DOLocationID__240', 'DOLocationID__241', 'DOLocationID__242', 'DOLocationID__243', 'DOLocationID__244', 'DOLocationID__245', 'DOLocationID__246', 'DOLocationID__247', 'DOLocationID__248', 'DOLocationID__249', 'DOLocationID__250', 'DOLocationID__251', 'DOLocationID__252', 'DOLocationID__253', 'DOLocationID__254', 'DOLocationID__255', 'DOLocationID__256', 'DOLocationID__257', 'DOLocationID__258', 'DOLocationID__259', 'DOLocationID__260', 'DOLocationID__261', 'DOLocationID__262', 'DOLocationID__263']
            locId_data = pd.DataFrame(columns=PULocIDs+DOLocIDs)

            for plid in PULocIDs:
                if 'PULocationID__' + str(int(p_zone_id)) == plid:
                    locId_data[plid] = [1]
                else:
                    locId_data[plid] = [0]

            for dlid in DOLocIDs:
                if 'DOLocationID__' + str(int(d_zone_id)) == dlid:
                    locId_data[dlid] = [1]
                else:
                    locId_data[dlid] = [0]

            locId_data = locId_data[['PULocationID__1','PULocationID__3','PULocationID__4','PULocationID__6','PULocationID__7','PULocationID__8','PULocationID__9','PULocationID__10','PULocationID__12','PULocationID__13','PULocationID__14','PULocationID__17','PULocationID__18','PULocationID__19','PULocationID__20','PULocationID__21','PULocationID__22','PULocationID__24','PULocationID__25','PULocationID__26','PULocationID__28','PULocationID__29','PULocationID__31','PULocationID__32','PULocationID__33','PULocationID__34','PULocationID__35','PULocationID__36','PULocationID__37','PULocationID__38','PULocationID__39','PULocationID__40','PULocationID__41','PULocationID__42','PULocationID__43','PULocationID__45','PULocationID__47','PULocationID__48','PULocationID__49','PULocationID__50','PULocationID__51','PULocationID__52','PULocationID__53','PULocationID__54','PULocationID__55','PULocationID__56','PULocationID__57','PULocationID__58','PULocationID__60','PULocationID__61','PULocationID__62','PULocationID__63','PULocationID__65','PULocationID__66','PULocationID__67','PULocationID__68','PULocationID__69','PULocationID__70','PULocationID__71','PULocationID__72','PULocationID__73','PULocationID__74','PULocationID__75','PULocationID__76','PULocationID__77','PULocationID__78','PULocationID__79','PULocationID__80','PULocationID__81','PULocationID__82','PULocationID__83','PULocationID__85','PULocationID__87','PULocationID__88','PULocationID__89','PULocationID__90','PULocationID__91','PULocationID__92','PULocationID__93','PULocationID__94','PULocationID__95','PULocationID__96','PULocationID__97','PULocationID__98','PULocationID__100','PULocationID__101','PULocationID__102','PULocationID__106','PULocationID__107','PULocationID__108','PULocationID__112','PULocationID__113','PULocationID__114','PULocationID__116','PULocationID__119','PULocationID__120','PULocationID__121','PULocationID__122','PULocationID__123','PULocationID__124','PULocationID__125','PULocationID__126','PULocationID__127','PULocationID__128','PULocationID__129','PULocationID__130','PULocationID__131','PULocationID__132','PULocationID__133','PULocationID__134','PULocationID__135','PULocationID__136','PULocationID__137','PULocationID__138','PULocationID__140','PULocationID__141','PULocationID__142','PULocationID__143','PULocationID__144','PULocationID__145','PULocationID__146','PULocationID__147','PULocationID__148','PULocationID__149','PULocationID__151','PULocationID__152','PULocationID__153','PULocationID__155','PULocationID__157','PULocationID__158','PULocationID__159','PULocationID__160','PULocationID__161','PULocationID__162','PULocationID__163','PULocationID__164','PULocationID__165','PULocationID__166','PULocationID__167','PULocationID__168','PULocationID__169','PULocationID__170','PULocationID__171','PULocationID__172','PULocationID__173','PULocationID__174','PULocationID__175','PULocationID__177','PULocationID__178','PULocationID__179','PULocationID__180','PULocationID__181','PULocationID__182','PULocationID__185','PULocationID__186','PULocationID__188','PULocationID__189','PULocationID__190','PULocationID__191','PULocationID__192','PULocationID__193','PULocationID__194','PULocationID__195','PULocationID__196','PULocationID__197','PULocationID__198','PULocationID__200','PULocationID__201','PULocationID__202','PULocationID__203','PULocationID__205','PULocationID__207','PULocationID__208','PULocationID__209','PULocationID__210','PULocationID__211','PULocationID__212','PULocationID__213','PULocationID__215','PULocationID__216','PULocationID__217','PULocationID__218','PULocationID__219','PULocationID__220','PULocationID__223','PULocationID__224','PULocationID__225','PULocationID__226','PULocationID__227','PULocationID__228','PULocationID__229','PULocationID__230','PULocationID__231','PULocationID__232','PULocationID__233','PULocationID__234','PULocationID__235','PULocationID__236','PULocationID__237','PULocationID__238','PULocationID__239','PULocationID__240','PULocationID__241','PULocationID__242','PULocationID__243','PULocationID__244','PULocationID__246','PULocationID__247','PULocationID__248','PULocationID__249','PULocationID__250','PULocationID__251','PULocationID__252','PULocationID__254','PULocationID__255','PULocationID__256','PULocationID__257','PULocationID__258','PULocationID__259','PULocationID__260','PULocationID__261','PULocationID__262','PULocationID__263','DOLocationID__1','DOLocationID__3','DOLocationID__4','DOLocationID__5','DOLocationID__6','DOLocationID__7','DOLocationID__8','DOLocationID__9','DOLocationID__10','DOLocationID__11','DOLocationID__12','DOLocationID__13','DOLocationID__14','DOLocationID__15','DOLocationID__16','DOLocationID__17','DOLocationID__18','DOLocationID__19','DOLocationID__20','DOLocationID__21','DOLocationID__22','DOLocationID__23','DOLocationID__24','DOLocationID__25','DOLocationID__26','DOLocationID__27','DOLocationID__28','DOLocationID__29','DOLocationID__30','DOLocationID__31','DOLocationID__32','DOLocationID__33','DOLocationID__34','DOLocationID__35','DOLocationID__36','DOLocationID__37','DOLocationID__38','DOLocationID__39','DOLocationID__40','DOLocationID__41','DOLocationID__42','DOLocationID__43','DOLocationID__44','DOLocationID__45','DOLocationID__46','DOLocationID__47','DOLocationID__48','DOLocationID__49','DOLocationID__50','DOLocationID__51','DOLocationID__52','DOLocationID__53','DOLocationID__54','DOLocationID__55','DOLocationID__56','DOLocationID__57','DOLocationID__58','DOLocationID__59','DOLocationID__60','DOLocationID__61','DOLocationID__62','DOLocationID__63','DOLocationID__64','DOLocationID__65','DOLocationID__66','DOLocationID__67','DOLocationID__68','DOLocationID__69','DOLocationID__70','DOLocationID__71','DOLocationID__72','DOLocationID__73','DOLocationID__74','DOLocationID__75','DOLocationID__76','DOLocationID__77','DOLocationID__78','DOLocationID__79','DOLocationID__80','DOLocationID__81','DOLocationID__82','DOLocationID__83','DOLocationID__84','DOLocationID__85','DOLocationID__86','DOLocationID__87','DOLocationID__88','DOLocationID__89','DOLocationID__90','DOLocationID__91','DOLocationID__92','DOLocationID__93','DOLocationID__94','DOLocationID__95','DOLocationID__96','DOLocationID__97','DOLocationID__98','DOLocationID__100','DOLocationID__101','DOLocationID__102','DOLocationID__106','DOLocationID__107','DOLocationID__108','DOLocationID__109','DOLocationID__111','DOLocationID__112','DOLocationID__113','DOLocationID__114','DOLocationID__115','DOLocationID__116','DOLocationID__117','DOLocationID__118','DOLocationID__119','DOLocationID__120','DOLocationID__121','DOLocationID__122','DOLocationID__123','DOLocationID__124','DOLocationID__125','DOLocationID__126','DOLocationID__127','DOLocationID__128','DOLocationID__129','DOLocationID__130','DOLocationID__131','DOLocationID__132','DOLocationID__133','DOLocationID__134','DOLocationID__135','DOLocationID__136','DOLocationID__137','DOLocationID__138','DOLocationID__139','DOLocationID__140','DOLocationID__141','DOLocationID__142','DOLocationID__143','DOLocationID__144','DOLocationID__145','DOLocationID__146','DOLocationID__147','DOLocationID__148','DOLocationID__149','DOLocationID__150','DOLocationID__151','DOLocationID__152','DOLocationID__153','DOLocationID__154','DOLocationID__155','DOLocationID__156','DOLocationID__157','DOLocationID__158','DOLocationID__159','DOLocationID__160','DOLocationID__161','DOLocationID__162','DOLocationID__163','DOLocationID__164','DOLocationID__165','DOLocationID__166','DOLocationID__167','DOLocationID__168','DOLocationID__169','DOLocationID__170','DOLocationID__171','DOLocationID__172','DOLocationID__173','DOLocationID__174','DOLocationID__175','DOLocationID__176','DOLocationID__177','DOLocationID__178','DOLocationID__179','DOLocationID__180','DOLocationID__181','DOLocationID__182','DOLocationID__183','DOLocationID__184','DOLocationID__185','DOLocationID__186','DOLocationID__187','DOLocationID__188','DOLocationID__189','DOLocationID__190','DOLocationID__191','DOLocationID__192','DOLocationID__193','DOLocationID__194','DOLocationID__195','DOLocationID__196','DOLocationID__197','DOLocationID__198','DOLocationID__200','DOLocationID__201','DOLocationID__202','DOLocationID__203','DOLocationID__204','DOLocationID__205','DOLocationID__206','DOLocationID__207','DOLocationID__208','DOLocationID__209','DOLocationID__210','DOLocationID__211','DOLocationID__212','DOLocationID__213','DOLocationID__214','DOLocationID__215','DOLocationID__216','DOLocationID__217','DOLocationID__218','DOLocationID__219','DOLocationID__220','DOLocationID__221','DOLocationID__222','DOLocationID__223','DOLocationID__224','DOLocationID__225','DOLocationID__226','DOLocationID__227','DOLocationID__228','DOLocationID__229','DOLocationID__230','DOLocationID__231','DOLocationID__232','DOLocationID__233','DOLocationID__234','DOLocationID__235','DOLocationID__236','DOLocationID__237','DOLocationID__238','DOLocationID__239','DOLocationID__240','DOLocationID__241','DOLocationID__242','DOLocationID__243','DOLocationID__244','DOLocationID__245','DOLocationID__246','DOLocationID__247','DOLocationID__248','DOLocationID__249','DOLocationID__250','DOLocationID__251','DOLocationID__252','DOLocationID__253','DOLocationID__254','DOLocationID__255','DOLocationID__256','DOLocationID__257','DOLocationID__258','DOLocationID__259','DOLocationID__260','DOLocationID__261','DOLocationID__262','DOLocationID__263','PULocationID__206','PULocationID__139','PULocationID__117','PULocationID__11','PULocationID__46','PULocationID__105','PULocationID__15','PULocationID__110','PULocationID__16','PULocationID__154','PULocationID__222','PULocationID__183']]
            pca = joblib.load('pca_zones.joblib')
            tmp_cls = ['PUDOLocIdPCA_{}'.format(x) for x in range(1, 131)]
            locId_data = pd.DataFrame(pca.transform(locId_data), columns=tmp_cls)

            df['day_avg_temperature'] = float(day_avg_temperature)

            df['day_precipitation'] = float(day_precipitation)

            pbr_dbr_l = ['pickup_borough__Bronx_x_dropoff_borough__Bronx', 'pickup_borough__Bronx_x_dropoff_borough__Brooklyn',
                         'pickup_borough__Bronx_x_dropoff_borough__Manhattan', 'pickup_borough__Bronx_x_dropoff_borough__Queens', 'pickup_borough__Brooklyn_x_dropoff_borough__Bronx', 'pickup_borough__Brooklyn_x_dropoff_borough__Brooklyn', 'pickup_borough__Brooklyn_x_dropoff_borough__EWR',
                         'pickup_borough__Brooklyn_x_dropoff_borough__Manhattan', 'pickup_borough__Brooklyn_x_dropoff_borough__Queens', 'pickup_borough__Brooklyn_x_dropoff_borough__Staten Island', 'pickup_borough__EWR_x_dropoff_borough__Brooklyn',
                         'pickup_borough__EWR_x_dropoff_borough__Manhattan', 'pickup_borough__Manhattan_x_dropoff_borough__Bronx', 'pickup_borough__Manhattan_x_dropoff_borough__Brooklyn', 'pickup_borough__Manhattan_x_dropoff_borough__EWR',
                         'pickup_borough__Manhattan_x_dropoff_borough__Manhattan', 'pickup_borough__Manhattan_x_dropoff_borough__Queens', 'pickup_borough__Manhattan_x_dropoff_borough__Staten Island', 'pickup_borough__Queens_x_dropoff_borough__Bronx',
                         'pickup_borough__Queens_x_dropoff_borough__Brooklyn', 'pickup_borough__Queens_x_dropoff_borough__EWR', 'pickup_borough__Queens_x_dropoff_borough__Manhattan', 'pickup_borough__Queens_x_dropoff_borough__Queens',
                         'pickup_borough__Queens_x_dropoff_borough__Staten Island', 'pickup_borough__Staten Island_x_dropoff_borough__Brooklyn', 'pickup_borough__Staten Island_x_dropoff_borough__Manhattan', 'pickup_borough__Staten Island_x_dropoff_borough__Staten Island']

            for pdbr in pbr_dbr_l:
                if 'pickup_borough__' + p_borough + '_x_dropoff_borough__' + d_borough == pdbr:
                    df[pdbr] = 1
                else:
                    df[pdbr] = 0

            df['PULocationID'] = int(p_zone_id)
            df['DOLocationID'] = int(d_zone_id)
            df['pickup_hour'] = float(p_hour)

            df = pd.DataFrame(df)
            df = pd.concat([df, locId_data], axis=1)

            target_enc_dict = joblib.load('target_enc_dict.joblib')
            cols_for_target_enc = ['PULocationID', 'DOLocationID', 'pickup_period_of_day', 'pickup_hour']
            for col in cols_for_target_enc:
                means = target_enc_dict[col + '_means']
                df[col + '_mean_encoded'] = df[col].map(means)
                medians = target_enc_dict[col + '_medians']
                df[col + '_median_encoded'] = df[col].map(medians)
                stds = target_enc_dict[col + '_stds']
                df[col + '_std_encoded'] = df[col].map(stds)
                mins = target_enc_dict[col + '_mins']
                df[col + '_min_encoded'] = df[col].map(mins)
                maxs = target_enc_dict[col + '_maxs']
                df[col + '_max_encoded'] = df[col].map(maxs)
                df.drop(col, axis=1, inplace=True)

            columns_ordered = ['passenger_count','pickup__is_weekend','day_avg_temperature','day_temperature_range','day_precipitation','is_passenger_count_gt_6','VendorID__1','VendorID__2','pickup_day_name__Saturday','pickup_day_name__Sunday','pickup_day_name__Wednesday','pickup_borough__Bronx','pickup_borough__Brooklyn','pickup_borough__EWR','pickup_borough__Manhattan','pickup_borough__Queens','pickup_borough__Staten Island','dropoff_borough__Bronx','dropoff_borough__Brooklyn','dropoff_borough__EWR','dropoff_borough__Manhattan','dropoff_borough__Queens','dropoff_borough__Staten Island','pickup_service_zone__Airports','pickup_service_zone__Boro Zone','pickup_service_zone__EWR','pickup_service_zone__Yellow Zone','dropoff_service_zone__Airports','dropoff_service_zone__Boro Zone','dropoff_service_zone__EWR','dropoff_service_zone__Yellow Zone','pickup_period_of_day__12_14','pickup_period_of_day__15_17','pickup_period_of_day__18_20','pickup_period_of_day__21_23','pickup_day_of_month__1','pickup_day_of_month__4','pickup_day_of_month__5','pickup_day_name__Thursday','pickup_day_name__Monday','PUDOLocIdPCA_1','PUDOLocIdPCA_2','PUDOLocIdPCA_3','PUDOLocIdPCA_4','PUDOLocIdPCA_5','PUDOLocIdPCA_6','PUDOLocIdPCA_7','PUDOLocIdPCA_8','PUDOLocIdPCA_9','PUDOLocIdPCA_10','PUDOLocIdPCA_11','PUDOLocIdPCA_12','PUDOLocIdPCA_13','PUDOLocIdPCA_14','PUDOLocIdPCA_15','PUDOLocIdPCA_16','PUDOLocIdPCA_17','PUDOLocIdPCA_18','PUDOLocIdPCA_19','PUDOLocIdPCA_20','PUDOLocIdPCA_21','PUDOLocIdPCA_22','PUDOLocIdPCA_23','PUDOLocIdPCA_24','PUDOLocIdPCA_25','PUDOLocIdPCA_26','PUDOLocIdPCA_27','PUDOLocIdPCA_28','PUDOLocIdPCA_29','PUDOLocIdPCA_30','PUDOLocIdPCA_31','PUDOLocIdPCA_32','PUDOLocIdPCA_33','PUDOLocIdPCA_34','PUDOLocIdPCA_35','PUDOLocIdPCA_36','PUDOLocIdPCA_37','PUDOLocIdPCA_38','PUDOLocIdPCA_39','PUDOLocIdPCA_40','PUDOLocIdPCA_41','PUDOLocIdPCA_42','PUDOLocIdPCA_43','PUDOLocIdPCA_44','PUDOLocIdPCA_45','PUDOLocIdPCA_46','PUDOLocIdPCA_47','PUDOLocIdPCA_48','PUDOLocIdPCA_49','PUDOLocIdPCA_50','PUDOLocIdPCA_51','PUDOLocIdPCA_52','PUDOLocIdPCA_53','PUDOLocIdPCA_54','PUDOLocIdPCA_55','PUDOLocIdPCA_56','PUDOLocIdPCA_57','PUDOLocIdPCA_58','PUDOLocIdPCA_59','PUDOLocIdPCA_60','PUDOLocIdPCA_61','PUDOLocIdPCA_62','PUDOLocIdPCA_63','PUDOLocIdPCA_64','PUDOLocIdPCA_65','PUDOLocIdPCA_66','PUDOLocIdPCA_67','PUDOLocIdPCA_68','PUDOLocIdPCA_69','PUDOLocIdPCA_70','PUDOLocIdPCA_71','PUDOLocIdPCA_72','PUDOLocIdPCA_73','PUDOLocIdPCA_74','PUDOLocIdPCA_75','PUDOLocIdPCA_76','PUDOLocIdPCA_77','PUDOLocIdPCA_78','PUDOLocIdPCA_79','PUDOLocIdPCA_80','PUDOLocIdPCA_81','PUDOLocIdPCA_82','PUDOLocIdPCA_83','PUDOLocIdPCA_84','PUDOLocIdPCA_85','PUDOLocIdPCA_86','PUDOLocIdPCA_87','PUDOLocIdPCA_88','PUDOLocIdPCA_89','PUDOLocIdPCA_90','PUDOLocIdPCA_91','PUDOLocIdPCA_92','PUDOLocIdPCA_93','PUDOLocIdPCA_94','PUDOLocIdPCA_95','PUDOLocIdPCA_96','PUDOLocIdPCA_97','PUDOLocIdPCA_98','PUDOLocIdPCA_99','PUDOLocIdPCA_100','PUDOLocIdPCA_101','PUDOLocIdPCA_102','PUDOLocIdPCA_103','PUDOLocIdPCA_104','PUDOLocIdPCA_105','PUDOLocIdPCA_106','PUDOLocIdPCA_107','PUDOLocIdPCA_108','PUDOLocIdPCA_109','PUDOLocIdPCA_110','PUDOLocIdPCA_111','PUDOLocIdPCA_112','PUDOLocIdPCA_113','PUDOLocIdPCA_114','PUDOLocIdPCA_115','PUDOLocIdPCA_116','PUDOLocIdPCA_117','PUDOLocIdPCA_118','PUDOLocIdPCA_119','PUDOLocIdPCA_120','PUDOLocIdPCA_121','PUDOLocIdPCA_122','PUDOLocIdPCA_123','PUDOLocIdPCA_124','PUDOLocIdPCA_125','PUDOLocIdPCA_126','PUDOLocIdPCA_127','PUDOLocIdPCA_128','PUDOLocIdPCA_129','PUDOLocIdPCA_130','pickup_borough__Bronx_x_dropoff_borough__Bronx','pickup_borough__Bronx_x_dropoff_borough__Brooklyn','pickup_borough__Bronx_x_dropoff_borough__Manhattan','pickup_borough__Bronx_x_dropoff_borough__Queens','pickup_borough__Brooklyn_x_dropoff_borough__Bronx','pickup_borough__Brooklyn_x_dropoff_borough__Brooklyn','pickup_borough__Brooklyn_x_dropoff_borough__EWR','pickup_borough__Brooklyn_x_dropoff_borough__Manhattan','pickup_borough__Brooklyn_x_dropoff_borough__Queens','pickup_borough__Brooklyn_x_dropoff_borough__Staten Island','pickup_borough__EWR_x_dropoff_borough__Brooklyn','pickup_borough__EWR_x_dropoff_borough__Manhattan','pickup_borough__Manhattan_x_dropoff_borough__Bronx','pickup_borough__Manhattan_x_dropoff_borough__Brooklyn','pickup_borough__Manhattan_x_dropoff_borough__EWR','pickup_borough__Manhattan_x_dropoff_borough__Manhattan','pickup_borough__Manhattan_x_dropoff_borough__Queens','pickup_borough__Manhattan_x_dropoff_borough__Staten Island','pickup_borough__Queens_x_dropoff_borough__Bronx','pickup_borough__Queens_x_dropoff_borough__Brooklyn','pickup_borough__Queens_x_dropoff_borough__EWR','pickup_borough__Queens_x_dropoff_borough__Manhattan','pickup_borough__Queens_x_dropoff_borough__Queens','pickup_borough__Queens_x_dropoff_borough__Staten Island','pickup_borough__Staten Island_x_dropoff_borough__Brooklyn','pickup_borough__Staten Island_x_dropoff_borough__Manhattan','pickup_borough__Staten Island_x_dropoff_borough__Staten Island','PULocationID_mean_encoded','PULocationID_median_encoded','PULocationID_std_encoded','PULocationID_min_encoded','PULocationID_max_encoded','DOLocationID_mean_encoded','DOLocationID_median_encoded','DOLocationID_std_encoded','DOLocationID_min_encoded','DOLocationID_max_encoded','pickup_period_of_day_mean_encoded','pickup_period_of_day_median_encoded','pickup_period_of_day_std_encoded','pickup_period_of_day_min_encoded','pickup_period_of_day_max_encoded','pickup_hour_mean_encoded','pickup_hour_median_encoded','pickup_hour_std_encoded','pickup_hour_min_encoded','pickup_hour_max_encoded']
            df = df[columns_ordered]

            # input_data = [
            #     [1.00000000e+00, 0.00000000e+00, 1.11111111e+01, 8.88888889e+00,
            #      9.65199950e-01, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
            #      5.71312188e-01, -3.93628480e-02, 2.35286809e-01, -4.91030564e-01,
            #      6.17834894e-01, 4.11953384e-01, 6.61002600e-01, 3.71806315e-01,
            #      -7.17083328e-03, 8.77905466e-03, 1.30886455e-01, -3.67388285e-02,
            #      -1.73419027e-01, -1.58233639e-01, -8.05321364e-02, 4.99761874e-02,
            #      -1.19281892e-01, 4.69293611e-02, -6.87803282e-02, -3.33865241e-03,
            #      8.24750096e-02, 7.85381997e-02, 9.41952770e-02, -9.83847758e-02,
            #      2.71350449e-03, 5.49472336e-03, -4.86722409e-03, 6.94749999e-03,
            #      -6.08729509e-02, 3.54108254e-02, 3.82588605e-02, -5.63975955e-02,
            #      -2.98860974e-02, -1.20845598e-03, 9.57282562e-03, 1.70083399e-02,
            #      -8.19726417e-03, 1.83190352e-02, 2.93429652e-02, -2.00212250e-02,
            #      9.96263831e-04, -1.06191423e-02, 1.54438523e-03, 1.17789985e-02,
            #      1.36832699e-02, -1.12194514e-03, -3.13229068e-02, -1.12244086e-02,
            #      -9.15797444e-03, -1.68661939e-03, -1.86579414e-02, 2.16885399e-03,
            #      1.27649409e-02, -9.92939375e-03, -2.71349134e-02, -1.88326061e-02,
            #      -3.54652741e-02, 1.02602051e-03, -1.10744052e-02, -8.92742397e-03,
            #      4.89049569e-03, -1.10975436e-02, -8.80165143e-03, 7.41446786e-03,
            #      -1.75467448e-03, 7.93750356e-03, -5.10338094e-03, -9.73457578e-03,
            #      5.98509182e-04, -3.64383588e-02, -7.53712731e-03, 9.14419734e-03,
            #      -2.59565713e-02, -1.79536599e-02, 4.85755232e-03, -4.02283338e-03,
            #      -2.86676462e-03, 3.98671191e-03, 1.36254121e-02, 1.12901408e-03,
            #      4.30032726e-03, -1.53335733e-02, -6.01585489e-03, -5.01707423e-03,
            #      7.67018264e-03, -2.91682575e-04, -2.48292068e-03, 2.25298771e-03,
            #      1.30480903e-03, 1.36284570e-03, -8.50015307e-04, -8.86906008e-04,
            #      9.57581956e-03, 2.66764051e-03, 5.18160591e-03, 6.99177887e-04,
            #      4.06118295e-03, 9.54449826e-04, 1.05247172e-03, -1.18004968e-04,
            #      -1.25701755e-03, 2.88597391e-03, 3.36219116e-03, 2.86948878e-03,
            #      1.83793951e-03, -6.89282472e-03, 3.78747227e-03, -4.31939825e-03,
            #      4.09785805e-04, -1.80220095e-03, 1.19832382e-03, -1.16937306e-04,
            #      6.78889659e-04, 4.95291715e-04, -2.28549987e-03, 2.20843380e-03,
            #      4.57864126e-07, -3.21461935e-03, -7.12390280e-04, -1.21207977e-03,
            #      -3.83542397e-04, -1.46672948e-03, -1.68226607e-04, -1.07474164e-04,
            #      -8.90840190e-05, -8.81821615e-04, 3.20316622e-04, 2.36385092e-04,
            #      1.75248205e-04, -5.33924973e-04, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            #      0.00000000e+00, 1.03670740e+01, 8.03333333e+00, 7.88480472e+00,
            #      6.83333333e-01, 8.43166667e+01, 1.02796970e+01, 8.08333333e+00,
            #      7.79721630e+00, 6.66666667e-01, 1.59700000e+02, 1.43211354e+01,
            #      1.11833333e+01, 1.16563912e+01, 5.16666667e-01, 1.77800000e+02,
            #      1.39098443e+01, 1.10000000e+01, 1.11353741e+01, 5.16666667e-01,
            #      1.63916667e+02]
            # ]
            # light_pred = light.predict(pd.DataFrame(input_data))

            light_pred = light.predict(df)
            dtest = xgboost.DMatrix(df)
            xgb_pred = xgb.predict(dtest)

            scaler = joblib.load('scaler_meta_svr.joblib')
            stacked_test_predictions_scaled = scaler.transform(np.column_stack((light_pred, xgb_pred)))
            ts_svr_pred = svr_meta_model.predict(stacked_test_predictions_scaled)

            stacked_test_predictions = np.column_stack((light_pred, xgb_pred))
            ts_light_pred = light_meta_model.predict(stacked_test_predictions)

            ts_pred = ((ts_svr_pred + ts_light_pred) / 2)[0]

            seconds = int((ts_pred % int(ts_pred)) * 60)
            minutes = int(ts_pred)

            resp = "Estimated duration = {} minutes and {} seconds".format(minutes, seconds)

            return render_template('predictor.html', pred=resp)
    except Exception as e:
        return render_template('predictor.html', pred=str(e))
    else:
        return render_template('predictor.html')


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
