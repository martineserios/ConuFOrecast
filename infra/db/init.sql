CREATE TABLE "models" (
  "model_id" varchar NOT NULL,
  "model_name" varchar,
  "model_version" varchar,
  "flow_units" varchar,
  "infiltration" varchar,
  "flow_routing" varchar,
  "link_offsets" varchar,
  "min_slope" float,
  "allow_ponding" boolean,
  "skip_steady_state" boolean,
  "start_date" date,
  "start_time" time,
  "report_start_date" date,
  "report_start_time" time,
  "end_date" date,
  "end_time" time,
  "sweep_start" date,
  "sweep_end" date,
  "report_step" time,
  "wet_step" time,
  "dry_step" time,
  "routing_step" time,
  "inertial_damping" varchar,
  "normal_flow_limited" varchar,
  "force_main_equation" varchar,
  "variable_step" float,
  "lengthening_step" integer,
  "min_surfarea" float,
  "max_trials" integer,
  "head_tolerance" float,
  "sys_flow" integer,
  "lat_flow_tol" integer,
  "minimum_step" float,
  "threads" integer,

  PRIMARY KEY ("model_id")
  -- FOREIGN KEY ("model_id") REFERENCES events ("model_id")
);

CREATE TABLE "nodes_coordinates" (
  "model_id" varchar NOT NULL,
  "node_id" varchar NOT NULL,
  "x_coord" float,
  "y_coord" float,
  "lat" float,
  "lon" float,

  PRIMARY KEY ("model_id", "node_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "subcatchments" (
  "model_id" varchar NOT NULL,
  "subcatchment_id" varchar NOT NULL,
  "raingage_id" varchar,
  "outlet" varchar,
  "area" float,
  "imperv" float,
  "width" float,
  "slope" float,
  "curb_len" float,
  -- "snow_pack" float,

  PRIMARY KEY ("model_id", "subcatchment_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
  );

-- CREATE TABLE "files" (
--   "model_id" varchar NOT NULL,
--   "interfacing_fails" varchar
-- );weirs

CREATE TABLE "evaporation" (
  "model_id" varchar NOT NULL,
  "data_source" varchar,
  "parameters" varchar,

  PRIMARY KEY ("model_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "links_conduits" (
  "model_id" varchar NOT NULL,
  "conduit_id" varchar NOT NULL,
  "from_node" varchar,
  "to_node" varchar,
  "length" float,
  "roughness" float,
  "in_offset" float,
  "out_offset" float,
  "init_flow" float,
  "max_flow" float,

  PRIMARY KEY ("model_id", "conduit_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")

);

CREATE TABLE "links_orifices" (
  "model_id" varchar NOT NULL,
  "orifice_id" varchar NOT NULL,
  "from_node" varchar,
  "to_node" varchar,
  "type" varchar,
  "offset" float,
  "q_coeff" float,
  "gated" boolean,
  "close_time" float,

  PRIMARY KEY ("model_id", "orifice_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "links_weirs" (
  "model_id" varchar NOT NULL,
  "weir_id" varchar NOT NULL,
  "from_node" varchar,
  "to_node" varchar,
  "type" varchar,
  "crest_ht" float,
  "q_coeff" float,
  "gated" boolean,
  "end_con" float,
  "end_coeff" float,
  "surcharge" boolean,
  -- "road_width" float,
  -- "road_surf" float,

  PRIMARY KEY ("model_id", "weir_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "xsections" (
  "model_id" varchar NOT NULL,
  "conduit_id" varchar,
  "shape" varchar,
  "geom_1" varchar,
  "geom_2" varchar,
  "geom_3" varchar,
  "geom_4" varchar,
  "barrels" varchar,
  -- "culvert" varchar,

  PRIMARY KEY ("model_id", "conduit_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

-- CREATE TABLE "transects" (
--   "transect_id" varchar
-- );

CREATE TABLE "losses" (
  "model_id" varchar NOT NULL,
  "link_id" varchar NOT NULL,
  "k_entry" float,
  "k_exit" float,
  "k_avg" float,
  "flap_gate" boolean,
  "seepage" float,

  PRIMARY KEY ("model_id", "link_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "subareas" (
  "model_id" varchar NOT NULL,
  "subcatchment_id" varchar NOT NULL,
  "n_imperv" float,
  "n_perv" float,
  "s_imperv" float,
  "s_perv" float,
  "pct_zero" float,
  "route_to" varchar,
  -- "pct_routed" integer,

  PRIMARY KEY ("model_id", "subcatchment_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

-- CREATE TABLE "curves" (
--   "model_id" varchar NOT NULL,
--   "storage_id" varchar NOT NULL,
--   "type" varchar,
--   "x_value" float,
--   "y_value" float,
--   PRIMARY KEY ("model_id", "storage_id")
-- );

CREATE TABLE "coorrdinates" (
  "model_id" varchar NOT NULL,
  "node_id" varchar NOT NULL,
  "x_coord" float,
  "y_coord" float,

  PRIMARY KEY ("model_id", "node_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

-- CREATE TABLE "map" (
--   "model_id" varchar NOT NULL,
--   "dimensions" varchar,
--   "units" varchar,
--   PRIMARY KEY ("model_id")
-- );

CREATE TABLE "vertices" (
  "model_id" varchar NOT NULL,
  "link_id" varchar NOT NULL,
  "x_coord" float,
  "y_coord" float,

  PRIMARY KEY ("model_id", "link_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "polygons" (
  "model_id" varchar NOT NULL,
  "subcatchment_id" varchar NOT NULL,
  "x_coord" float,
  "y_coord" float,

  PRIMARY KEY ("model_id", "subcatchment_id"),
  FOREIGN KEY ("model_id","subcatchment_id") REFERENCES subcatchments ("model_id","subcatchment_id")
);

-- CREATE TABLE "profiles" (
--   "model_id" varchar NOT NULL,
--   "profile_id" varchar NOT NULL,
--   "links" varchar,
--   PRIMARY KEY ("model_id", "profile_id")
-- );

-- CREATE TABLE "symbols" (
--   "model_id" varchar NOT NULL,
--   "raingage_id" varchar NOT NULL,
--   "x_coord" float,
--   "y_coord" float,
--   PRIMARY KEY ("model_id", "raingage_id")
-- );

CREATE TABLE "infiltration" (
  "model_id" varchar NOT NULL,
  "subcatchment_id" varchar NOT NULL,
  "max_rate" float,
  "min_rate" float,
  "decay" float,
  "dry_time" float,
  "max_infil" float,

  PRIMARY KEY ("model_id", "subcatchment_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "nodes_junctions" (
  "model_id" varchar NOT NULL,
  "junction_id" varchar NOT NULL,
  "elevation" float,
  "max_depth" float,
  "init_depth" float,
  "sur_depth" float,
  "aponded" float,

  PRIMARY KEY ("model_id", "junction_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "nodes_outfalls" (
  "model_id" varchar NOT NULL,
  "outfall_id" varchar,
  "elevation" float,
  "type" varchar,
  -- "stage_data" integer,
  "gated" varchar,
  -- "route_to" varchar,

  PRIMARY KEY ("model_id", "outfall_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);

CREATE TABLE "nodes_storage" (
  "model_id" varchar NOT NULL,
  "storage_id" varchar NOT NULL,
  "elevation" float,
  "max_depth" float,
  "init_depth" float,
  "shape" varchar,
  "curve_name_params" varchar,
  "n_a" float,
  "f_evap" float,
  -- "psi" float,
  -- "k_sat" float,
  -- "imd" float,

  PRIMARY KEY ("model_id", "storage_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id")
);


CREATE TABLE "precipitation_event" (
  "precipitation_id" varchar NOT NULL,
  
  PRIMARY KEY ("precipitation_id")
);


CREATE TABLE "raingages_metadata" (
  "precipitation_id" varchar NOT NULL,
  "raingage_id" varchar NOT NULL,
  "interval" time,
  "format" varchar,
  "unit" varchar,

  PRIMARY KEY ("precipitation_id","raingage_id"),
  FOREIGN KEY ("precipitation_id") REFERENCES precipitation_event ("precipitation_id")
);

CREATE TABLE "raingages_timeseries" (
  "precipitation_id" varchar NOT NULL,
  "raingage_id" varchar NOT NULL,
  "elapsed_time" timestamp NOT NULL,
  "value" float,

  PRIMARY KEY ("precipitation_id","raingage_id", "elapsed_time"),
  FOREIGN KEY ("precipitation_id","raingage_id") REFERENCES raingages_metadata ("precipitation_id", "raingage_id")
);


CREATE TABLE "events" (
  "event_id" varchar NOT NULL,
  "model_id" varchar NOT NULL,
  "precipitation_id" varchar NOT NULL,

  PRIMARY KEY ("event_id"),
  FOREIGN KEY ("model_id") REFERENCES models ("model_id"),
  FOREIGN KEY ("precipitation_id") REFERENCES precipitation_event ("precipitation_id")  
);


CREATE TABLE "events_subcatchments" (
  "event_id" varchar NOT NULL,
  "elapsed_time" timestamp NOT NULL,
  "subcatchment_id" varchar NOT NULL,
  "rainfall" float,
  "snow_depth" float,
  "evaporation_loss" float,
  "infiltration_loss" float,
  "runoff_rate" float,
  "groundwater_outflow" float,
  "groundwater_elevation" float,
  "soil_moisture" float,

  PRIMARY KEY ("event_id", "subcatchment_id", "elapsed_time"),
  FOREIGN KEY ("event_id") REFERENCES events ("event_id")
);

CREATE TABLE "events_nodes" (
  "event_id" varchar NOT NULL,
  "elapsed_time" timestamp NOT NULL,
  "node_id" varchar NOT NULL,
  "depth_above_invert" float,
  "hydraulic_head" float,
  "volume_stored_ponded" float,
  "lateral_inflow" float,
  "total_inflow" float,
  "flow_lost_flooding" float,

  PRIMARY KEY ("event_id", "node_id", "elapsed_time"),
  FOREIGN KEY ("event_id") REFERENCES events ("event_id")
);

CREATE TABLE "events_links" (
  "event_id" varchar NOT NULL,
  "elapsed_time" timestamp NOT NULL,
  "link_id" varchar NOT NULL,
  "flow_rate" float,
  "flow_depth" float,
  "flow_velocity" float,
  "froude_number" float,
  "capacity" float,

  PRIMARY KEY ("event_id", "link_id", "elapsed_time"),
  FOREIGN KEY ("event_id") REFERENCES events ("event_id")
);

CREATE TABLE "events_systems" (
  "event_id" varchar NOT NULL,
  "elapsed_time" timestamp NOT NULL,
  "system_id" varchar NOT NULL,
  "air_tempreture" float,
  "rainfall" float,
  "runoff" float,
  "dry_weather_inflow" float,
  "groundwater_inflow" float,
  "rdii_inflow" float,
  "user_direct_inflow" float,
  "total_lateral_inflow" float,
  "flow_lost_to_flooding" float,
  "flow_leaving_outfalls" float,
  "volume_stored_water" float,
  "evaporation_rate" float,
  "potential_pet" float,

  PRIMARY KEY ("event_id", "system_id", "elapsed_time"),
  FOREIGN KEY ("event_id") REFERENCES events ("event_id")
);

CREATE TABLE "events_pollutants" (
  "event_id" varchar NOT NULL,
  "elapsed_time" timestamp NOT NULL,
  "pollutant_id" int NOT NULL,

  PRIMARY KEY ("event_id", "pollutant_id", "elapsed_time"),
  FOREIGN KEY ("event_id") REFERENCES events ("event_id")
);