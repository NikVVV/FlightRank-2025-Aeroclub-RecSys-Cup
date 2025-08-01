companyID - Frequency (count) encoding - delete
corporateTariffCode - tariff_code_filled, fill nulls, astype(cat) - delete
frequentFlyer - frequentFlyer_n_programs, 
nationality - nationality_cat - delete
miniRules0_statusInfos - miniRules0_statusInfos_cat, 3 categories - delete 
miniRules1_statusInfos - miniRules1_statusInfos_cat, 3 categories - delete 
pricingInfo_isAccessTP - pricingInfo_isAccessTP_cat, 3 cat - delete
pricingInfo_passengerCount - only 1 value - delete
profileId/ranker_id - unique_ranker_count,,,
requestDate - wday_sin, wday_cos, hour_sin, hour_cos - delete  
searchRoute - oneway_route, координаты городов или аэропортов, подсчитано расстояние между городами - delete
sex - так оставляем
taxes - так оставляем
totalPrice - цена/средняя цена билета с тем же направлением, цена/средняя цена на запрашиваемую дату по тем же направлениям
miniRules1_percentage
legs0_duration - перевели в часы (ебана рот это заняло час)
legs1_duration - перевели в часы (ебана рот это заняло час)
legs{}_segments{}_aircraft_code - частотный энкодер 
legs{}_segments{}_baggageAllowance_weightMeasurementType, legs{}_segments{}_baggageAllowance_quantity - разделил на две колонки, кг и места
_arrivalTo_airport_iata/_departureFrom_airport_iata - подсчитать количество смен аэропортов



1) Количество часов на пересадки layover_hours_leg0, layover_hours_leg1
2) Суммарное количество времени на пересадки layover_hours_leg0, layover_hours_leg1
3) Время до полета (в часах) days_before_flight_leg0
4) Аэропорт прилета отличается от аэропорта вылета legs{leg}_airport_changes_count
5) ночная пересадка      night_layover_leg0, night_layover_leg1
6) смена аэропорта  legs{leg}_airport_changes_count
7) Одна ли авиакомпания перевозчик  same_operator_carrier_leg0, same_operator_carrier_leg1
8) входит ли билет в программу лояльности клиента       ticket_is_in_FFprogramms_leg0, ticket_is_in_FFprogramms_leg1
9) ранг по цене внтури сессии        totalPrice_rank
10) ранг по времени перелета внутри сессии      totalTime_hours_ranked
11) количество авиакомпаний перевозчиков    same_operator_carrier_leg0, same_operator_carrier_leg1
12) разные ли условия по багажу среди всего перелета baggage_kg_equal_flag, baggage_units_equal_flag
13) количество вариантов внутри сессии tickets_in_session
14) ранг внутри сессии на количество оставшихся билетов remainingTickets_avg, remainingTickets_rank
15) заменить profileID, на частоту насколько часто человек покупает билеты user_search_freq
16) В каком количестве перелетов совпадает оператор и продавец по компаниям  operator_marketer_match_rate
17) Стоимость перелета в пределах 20% от самого дешевого варианта within_20pct_of_min
18) День недели для вылета leg0_depday_sin, leg1_depday_sin, leg0_depday_cos, leg1_depday_cos
19) день недели прилета leg0_arrday_sin, leg1_arrday_sin, leg0_arrday_cos, leg1_arrday_cos
20) оптимальный билет(по стоимости и времени. (стоимость/опт.стоимость + время/опт время) / 2 ) opt_ticket_score



21)  Количестве сегментов n_segments_leg{leg}
22) is_one_way
23) is_direct_leg0, is_direct_leg1
24) price_per_tax
25) tax_rate
26) log_price
27) duration_ratio
28) has_corporate_tariff
29) has_access_tp
30) total_fees
31) is_popular_route
32) avg_cabin_class
33) cabin_class_diff
34) n_ff_programs
35) both_direct
36) is_vip_freq
37) has_fees
38) fee_rate
39) price_pct_rank
40) is_cheapest
41) price_from_median
42) is_min_segments
43) is_direct_cheapest







21) Как часто компания выбирает эту компанию-перевозчика(на leg0 и leg1, seg0). 
22) Популярность конкретной авиакомпании (насколько часто их билет выбирали среди прочих)


Топ фичи
Added               opt_ticket_score  →  hit@3 = 0.5022
Added      miniRules1_monetaryAmount  →  hit@3 = 0.5263
Added               avg_oneway_price  →  hit@3 = 0.5383
Added                      log_price  →  hit@3 = 0.5562
Added                   company_freq  →  hit@3 = 0.5749
Added       legs0_departureAt_period  →  hit@3 = 0.5813
Added       legs1_departureAt_period  →  hit@3 = 0.5962