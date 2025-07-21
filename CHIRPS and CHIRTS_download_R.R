# Set working directory to the script's directory for robust file/folder creation
this_file <- tryCatch({
  if (interactive()) {
    dirname(normalizePath(sys.frames()[[1]]$ofile %||% getSrcDirectory(function(x) x)))
  } else {
    dirname(normalizePath(commandArgs(trailingOnly = FALSE)[grep('--file=', commandArgs(), value = TRUE)]))
  }
}, error = function(e) NULL)
if (!is.null(this_file) && this_file != "") setwd(this_file)

# Set up tempdir-based logs and cache directories
.temp_base <- normalizePath(tempdir(), winslash = "\\", mustWork = TRUE)
logs_dir <- file.path(.temp_base, "logs")
chirps_cache_dir <- file.path(.temp_base, "chirps_cache")
chirts_cache_dir <- file.path(.temp_base, "chirts_cache")
if (!dir.exists(logs_dir)) dir.create(logs_dir, showWarnings = FALSE, recursive = TRUE)
if (!dir.exists(chirps_cache_dir)) dir.create(chirps_cache_dir, showWarnings = FALSE, recursive = TRUE)
if (!dir.exists(chirts_cache_dir)) dir.create(chirts_cache_dir, showWarnings = FALSE, recursive = TRUE)

# Code for downloading CHIRPS precipitation and CHIRTS temperature data
# Fixed %...>% error by adding promises package and improved Excel validation
# Single date range and added yearly reports
# by: James Albert Kaunang (original)
# Last updated: 2025-05-21

library(chirps)
library(terra)
library(writexl)
library(readxl)
library(lubridate)
library(dplyr)
library(beepr)
library(logging)
library(shiny)
library(jsonlite)
library(future)
library(promises)
library(ggplot2)
library(digest)

# Suppress dplyr NSE variable warnings
utils::globalVariables(c(
  "tmin", "tmax", "rhum", "heatindex", "chirps", "lon", "lat", "year", "month", "day", "Name", "Longitude", "Latitude", "date",
  "monthly_total_precip", "monthly_mean_tmax", "monthly_mean_tmin", "monthly_mean_rhum", "monthly_mean_heatindex",
  "yearly_total_precip", "yearly_mean_tmax", "yearly_mean_tmin", "yearly_mean_rhum", "yearly_mean_heatindex",
  "max_precip", "min_precip", "max_precip_date", "min_precip_date", "monthly_max", "total_precip", "yearly_max",
  "max_tmax", "min_tmax", "max_tmin", "min_tmin", "max_rhum", "min_rhum", "max_heatindex", "min_heatindex",
  "max_tmax_date", "min_tmax_date", "max_tmin_date", "min_tmin_date", "max_rhum_date", "min_rhum_date", "max_heatindex_date", "min_heatindex_date",
  "Value", "Variable"
))

# Function to validate inputs
validate_inputs <- function(coord_file, dates) {
  if (is.null(coord_file)) stop("No coordinate file provided.")
  coord <- read_excel(coord_file)
  required_cols <- c("Longitude", "Latitude", "Name")
  
  # Check for required columns
  missing_cols <- setdiff(required_cols, colnames(coord))
  if (length(missing_cols) > 0) {
    stop("Coordinate file missing required columns: ", paste(missing_cols, collapse = ", "))
  }
  
  # Select only required columns and warn about extras
  extra_cols <- setdiff(colnames(coord), required_cols)
  if (length(extra_cols) > 0) {
    warning("Extra columns in coordinate file ignored: ", paste(extra_cols, collapse = ", "))
  }
  coord <- coord[, required_cols, drop = FALSE]
  
  # Validate data
  if (nrow(coord) == 0) stop("Coordinate file is empty.")
  if (any(is.na(coord$Longitude) | is.na(coord$Latitude) | is.na(coord$Name))) {
    stop("Coordinate file contains missing values in Longitude, Latitude, or Name.")
  }
  
  # Validate dates
  if (length(dates) != 2 || !all(!is.na(as.Date(dates)))) {
    stop("Invalid date format in date range: ", paste(dates, collapse = " to "))
  }
  
  return(coord)
}

# Function to download CHIRPS data with retry logic and caching
# Added progress callback for granular tracking
# Now uses cache in tempdir to reduce API calls

download_chirps_data <- function(lonlat, dates, server = "ClimateSERV", retries = 3, progress = NULL) {
  cache_file <- file.path(chirps_cache_dir, paste0(
    "chirps_",
    digest::digest(list(lonlat, dates, server)),
    ".rds"
  ))
  if (file.exists(cache_file)) {
    if (!is.null(progress)) progress("Loading CHIRPS data from cache...")
    return(readRDS(cache_file))
  }
  for (i in 1:retries) {
    if (!is.null(progress)) progress(paste0("Downloading CHIRPS attempt ", i, "..."))
    result <- tryCatch({
      data <- get_chirps(lonlat, dates, server = server)
      if (nrow(data) > 0) {
        if (!is.null(progress)) progress("CHIRPS download successful.")
        saveRDS(data, cache_file)
        return(data)
      } else {
        stop("Empty data returned")
      }
    }, error = function(e) {
      if (!is.null(progress)) progress(paste0("CHIRPS download failed: ", e$message))
      if (i == retries) return(NULL)
      Sys.sleep(2)
      NULL
    })
    if (!is.null(result)) return(result)
  }
  showNotification("Failed to download CHIRPS data after retries", type = "error")
  if (!is.null(progress)) progress("CHIRPS download failed after retries.")
  return(NULL)
}

# Function to download CHIRTS data (Tmax, Tmin, RHum, HeatIndex) with caching
# Added progress callback for granular tracking
# Now uses cache in tempdir to reduce API calls

download_chirts_data <- function(lonlat, dates, var = "All", retries = 3, progress = NULL) {
  cache_file <- file.path(chirts_cache_dir, paste0(
    "chirts_", var, "_", digest::digest(list(lonlat, dates, var)), ".rds"
  ))
  if (file.exists(cache_file)) {
    if (!is.null(progress)) progress("Loading CHIRTS data from cache...")
    return(readRDS(cache_file))
  }
  date_seq <- seq.Date(as.Date(dates[1]), as.Date(dates[2]), by = "day")
  grid <- data.frame(lon = lonlat$lon[1], lat = lonlat$lat[1], date = date_seq)
  tmax <- tmin <- rhum <- heatindex <- NULL
  if (var == "All" || var == "Tmax") {
    if (!is.null(progress)) progress("Downloading CHIRTS Tmax...")
    tmax <- NULL
    for (i in 1:retries) {
      tmax <- tryCatch({
        data <- get_chirts(lonlat, dates, var = "Tmax")
        if (nrow(data) > 0) {
          colnames(data)[tolower(colnames(data)) == "chirts"] <- "tmax"
          data[, c("date", "tmax")]
        } else stop("Empty data returned")
      }, error = function(e) {
        if (!is.null(progress)) progress(paste0("Tmax download failed: ", e$message))
        if (i == retries) return(NULL)
        Sys.sleep(2)
        NULL
      })
      if (!is.null(tmax)) break
    }
    if (!is.null(progress)) progress("Tmax download complete.")
  }
  if (var == "All" || var == "Tmin") {
    if (!is.null(progress)) progress("Downloading CHIRTS Tmin...")
    tmin <- NULL
    for (i in 1:retries) {
      tmin <- tryCatch({
        data <- get_chirts(lonlat, dates, var = "Tmin")
        if (nrow(data) > 0) {
          colnames(data)[tolower(colnames(data)) == "chirts"] <- "tmin"
          data[, c("date", "tmin")]
        } else stop("Empty data returned")
      }, error = function(e) {
        if (!is.null(progress)) progress(paste0("Tmin download failed: ", e$message))
        if (i == retries) return(NULL)
        Sys.sleep(2)
        NULL
      })
      if (!is.null(tmin)) break
    }
    if (!is.null(progress)) progress("Tmin download complete.")
  }
  if (var == "All" || var == "RHum") {
    if (!is.null(progress)) progress("Downloading CHIRTS RHum...")
    rhum <- NULL
    for (i in 1:retries) {
      rhum <- tryCatch({
        data <- get_chirts(lonlat, dates, var = "RHum")
        if (nrow(data) > 0) {
          colnames(data)[tolower(colnames(data)) == "chirts"] <- "rhum"
          data[, c("date", "rhum")]
        } else stop("Empty data returned")
      }, error = function(e) {
        if (!is.null(progress)) progress(paste0("RHum download failed: ", e$message))
        if (i == retries) return(NULL)
        Sys.sleep(2)
        NULL
      })
      if (!is.null(rhum)) break
    }
    if (!is.null(progress)) progress("RHum download complete.")
  }
  if (var == "All" || var == "HeatIndex") {
    if (!is.null(progress)) progress("Downloading CHIRTS HeatIndex...")
    heatindex <- NULL
    for (i in 1:retries) {
      heatindex <- tryCatch({
        data <- get_chirts(lonlat, dates, var = "HeatIndex")
        if (nrow(data) > 0) {
          colnames(data)[tolower(colnames(data)) == "chirts"] <- "heatindex"
          data[, c("date", "heatindex")]
        } else stop("Empty data returned")
      }, error = function(e) {
        if (!is.null(progress)) progress(paste0("HeatIndex download failed: ", e$message))
        if (i == retries) return(NULL)
        Sys.sleep(2)
        NULL
      })
      if (!is.null(heatindex)) break
    }
    if (!is.null(progress)) progress("HeatIndex download complete.")
  }
  all_data <- grid
  if (!is.null(tmax)) all_data <- merge(all_data, tmax, by = "date", all.x = TRUE)
  if (!is.null(tmin)) all_data <- merge(all_data, tmin, by = "date", all.x = TRUE)
  if (!is.null(rhum)) all_data <- merge(all_data, rhum, by = "date", all.x = TRUE)
  if (!is.null(heatindex)) all_data <- merge(all_data, heatindex, by = "date", all.x = TRUE)
  expected_cols <- c("tmax", "tmin", "rhum", "heatindex")
  for (v in expected_cols) {
    if (!v %in% colnames(all_data)) {
      all_data[[v]] <- NA_real_
    }
  }
  all_data <- all_data[, c("lon", "lat", "date", expected_cols), drop = FALSE]
  if (nrow(all_data) == 0 || all(is.na(all_data[, expected_cols]))) {
    showNotification("No valid CHIRTS data downloaded for any variable", type = "error")
    if (!is.null(progress)) progress("CHIRTS download failed for all variables.")
    return(NULL)
  }
  saveRDS(all_data, cache_file)
  if (!is.null(progress)) progress("CHIRTS download and merge complete.")
  return(all_data)
}

# Function to process climate data (CHIRPS or CHIRTS)
# Added progress callback for granular tracking
process_climate_data <- function(data, type = "CHIRPS", progress = NULL) {
  if (!is.null(progress)) progress("Processing: extracting year, month, day...")
  data$year <- year(data$date)
  data$month <- month(data$date)
  data$day <- day(data$date)
  if (type == "CHIRPS") {
    if (!is.null(progress)) progress("Processing: summarising CHIRPS monthly totals...")
    monthly_totals <- data %>%
      group_by(lon, lat, year, month) %>%
      summarise(monthly_total_precip = sum(chirps, na.rm = TRUE), .groups = "drop")
    if (!is.null(progress)) progress("Processing: summarising CHIRPS yearly totals...")
    yearly_totals <- data %>%
      group_by(lon, lat, year) %>%
      summarise(yearly_total_precip = sum(chirps, na.rm = TRUE), .groups = "drop")
  } else {
    for (col in c("tmax", "tmin", "rhum", "heatindex")) {
      if (!col %in% names(data)) data[[col]] <- NA_real_
    }
    if (!is.null(progress)) progress("Processing: summarising CHIRTS monthly means...")
    monthly_totals <- data %>%
      group_by(lon, lat, year, month) %>%
      summarise(
        monthly_mean_tmax = mean(tmax, na.rm = TRUE),
        monthly_mean_tmin = mean(tmin, na.rm = TRUE),
        monthly_mean_rhum = mean(rhum, na.rm = TRUE),
        monthly_mean_heatindex = mean(heatindex, na.rm = TRUE),
        .groups = "drop"
      )
    if (!is.null(progress)) progress("Processing: summarising CHIRTS yearly means...")
    yearly_totals <- data %>%
      group_by(lon, lat, year) %>%
      summarise(
        yearly_mean_tmax = mean(tmax, na.rm = TRUE),
        yearly_mean_tmin = mean(tmin, na.rm = TRUE),
        yearly_mean_rhum = mean(rhum, na.rm = TRUE),
        yearly_mean_heatindex = mean(heatindex, na.rm = TRUE),
        .groups = "drop"
      )
  }
  if (!is.null(progress)) progress("Processing: done.")
  return(list(data = data, monthly_totals = monthly_totals, yearly_totals = yearly_totals))
}

# Function to generate report
# Added progress callback for granular tracking
generate_report <- function(data, coord, type = "CHIRPS", progress = NULL) {
  if (!is.null(progress)) progress("Generating report: monthly...")
  if (type == "CHIRPS") {
    monthly_report <- data %>%
      group_by(lon, lat, year, month) %>%
      summarise(
        max_precip = max(chirps, na.rm = TRUE),
        min_precip = ifelse(any(chirps > 1), min(chirps[chirps > 1], na.rm = TRUE), NA),
        max_precip_date = as.Date(date[which.max(chirps)]),
        min_precip_date = ifelse(any(chirps > 1), as.Date(date[which(chirps == min(chirps[chirps > 1], na.rm = TRUE))[1]]), NA),
        monthly_max = max(chirps, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(min_precip_date = as.Date(min_precip_date, origin = "1970-01-01"))
    if (!is.null(progress)) progress("Generating report: yearly...")
    yearly_report <- data %>%
      group_by(lon, lat, year) %>%
      summarise(
        total_precip = sum(chirps, na.rm = TRUE),
        max_precip = max(chirps, na.rm = TRUE),
        min_precip = ifelse(any(chirps > 1), min(chirps[chirps > 1], na.rm = TRUE), NA),
        max_precip_date = as.Date(date[which.max(chirps)]),
        min_precip_date = ifelse(any(chirps > 1), as.Date(date[which(chirps == min(chirps[chirps > 1], na.rm = TRUE))[1]]), NA),
        yearly_max = max(chirps, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(min_precip_date = as.Date(min_precip_date, origin = "1970-01-01"))
  } else {
    monthly_report <- data %>%
      group_by(lon, lat, year, month) %>%
      summarise(
        max_tmax = na_if(max(tmax, na.rm = TRUE), -Inf),
        min_tmax = na_if(min(tmax, na.rm = TRUE), Inf),
        max_tmin = na_if(max(tmin, na.rm = TRUE), -Inf),
        min_tmin = na_if(min(tmin, na.rm = TRUE), Inf),
        max_rhum = na_if(max(rhum, na.rm = TRUE), -Inf),
        min_rhum = na_if(min(rhum, na.rm = TRUE), Inf),
        max_heatindex = na_if(max(heatindex, na.rm = TRUE), -Inf),
        min_heatindex = na_if(min(heatindex, na.rm = TRUE), Inf),
        max_tmax_date = as.Date(date[which.max(tmax)]),
        min_tmax_date = as.Date(date[which.min(tmax)]),
        max_tmin_date = as.Date(date[which.max(tmin)]),
        min_tmin_date = as.Date(date[which.min(tmin)]),
        max_rhum_date = as.Date(date[which.max(rhum)]),
        min_rhum_date = as.Date(date[which.min(rhum)]),
        max_heatindex_date = as.Date(date[which.max(heatindex)]),
        min_heatindex_date = as.Date(date[which.min(heatindex)]),
        monthly_max_tmax = na_if(max(tmax, na.rm = TRUE), -Inf),
        monthly_max_tmin = na_if(max(tmin, na.rm = TRUE), -Inf),
        monthly_max_rhum = na_if(max(rhum, na.rm = TRUE), -Inf),
        monthly_max_heatindex = na_if(max(heatindex, na.rm = TRUE), -Inf),
        .groups = "drop"
      ) %>%
      mutate(
        max_tmax_date = as.Date(max_tmax_date, origin = "1970-01-01"),
        min_tmax_date = as.Date(min_tmax_date, origin = "1970-01-01"),
        max_tmin_date = as.Date(max_tmin_date, origin = "1970-01-01"),
        min_tmin_date = as.Date(min_tmin_date, origin = "1970-01-01"),
        max_rhum_date = as.Date(max_rhum_date, origin = "1970-01-01"),
        min_rhum_date = as.Date(min_rhum_date, origin = "1970-01-01"),
        max_heatindex_date = as.Date(max_heatindex_date, origin = "1970-01-01"),
        min_heatindex_date = as.Date(min_heatindex_date, origin = "1970-01-01")
      )
    if (!is.null(progress)) progress("Generating report: yearly...")
    yearly_report <- data %>%
      group_by(lon, lat, year) %>%
      summarise(
        mean_tmax = mean(tmax, na.rm = TRUE),
        mean_tmin = mean(tmin, na.rm = TRUE),
        mean_rhum = mean(rhum, na.rm = TRUE),
        mean_heatindex = mean(heatindex, na.rm = TRUE),
        max_tmax = na_if(max(tmax, na.rm = TRUE), -Inf),
        min_tmax = na_if(min(tmax, na.rm = TRUE), Inf),
        max_tmin = na_if(max(tmin, na.rm = TRUE), -Inf),
        min_tmin = na_if(min(tmin, na.rm = TRUE), Inf),
        max_rhum = na_if(max(rhum, na.rm = TRUE), -Inf),
        min_rhum = na_if(min(rhum, na.rm = TRUE), Inf),
        max_heatindex = na_if(max(heatindex, na.rm = TRUE), -Inf),
        min_heatindex = na_if(min(heatindex, na.rm = TRUE), Inf),
        max_tmax_date = as.Date(date[which.max(tmax)]),
        min_tmax_date = as.Date(date[which.min(tmax)]),
        max_tmin_date = as.Date(date[which.max(tmin)]),
        min_tmin_date = as.Date(date[which.min(tmin)]),
        max_rhum_date = as.Date(date[which.max(rhum)]),
        min_rhum_date = as.Date(date[which.min(rhum)]),
        max_heatindex_date = as.Date(date[which.max(heatindex)]),
        min_heatindex_date = as.Date(date[which.min(heatindex)]),
        yearly_max_tmax = na_if(max(tmax, na.rm = TRUE), -Inf),
        yearly_max_tmin = na_if(max(tmin, na.rm = TRUE), -Inf),
        yearly_max_rhum = na_if(max(rhum, na.rm = TRUE), -Inf),
        yearly_max_heatindex = na_if(max(heatindex, na.rm = TRUE), -Inf),
        .groups = "drop"
      ) %>%
      mutate(
        max_tmax_date = as.Date(max_tmax_date, origin = "1970-01-01"),
        min_tmax_date = as.Date(min_tmax_date, origin = "1970-01-01"),
        max_tmin_date = as.Date(max_tmin_date, origin = "1970-01-01"),
        min_tmin_date = as.Date(min_tmin_date, origin = "1970-01-01"),
        max_rhum_date = as.Date(max_rhum_date, origin = "1970-01-01"),
        min_rhum_date = as.Date(min_rhum_date, origin = "1970-01-01"),
        max_heatindex_date = as.Date(max_heatindex_date, origin = "1970-01-01"),
        min_heatindex_date = as.Date(min_heatindex_date, origin = "1970-01-01")
      )
  }
  if (!is.null(progress)) progress("Report generation done.")
  monthly_report <- monthly_report %>%
    left_join(select(coord, Name, Longitude, Latitude), by = c("lon" = "Longitude", "lat" = "Latitude"))
  yearly_report <- yearly_report %>%
    left_join(select(coord, Name, Longitude, Latitude), by = c("lon" = "Longitude", "lat" = "Latitude"))
  return(list(monthly = monthly_report, yearly = yearly_report))
}

# UI for the Shiny app
ui <- fluidPage(
  titlePanel("CHIRPS and CHIRTS Data Downloader"),
  sidebarLayout(
    sidebarPanel(
      fileInput("coord_file", "Choose Coordinate File", accept = c(".xlsx")),
      selectInput("data_type", "Data Type", choices = c("CHIRPS", "CHIRTS")),
      dateRangeInput("date_range", "Date Range", start = "2010-01-01", end = "2010-12-31"),
      # checkboxInput("save_raw_data", "Save Raw Data", value = FALSE),  # Removed as raw data is always included
      selectInput("output_format", "Output Format", choices = c("xlsx", "csv")),
      actionButton("run", "Run"),
      downloadButton("download_data_files", "Save Data"),
      actionButton("save_config", "Save Config"),
      fileInput("config_file", "Choose Config File", accept = c(".json")),
      uiOutput("coord_selector"),
      actionButton("clear_cache", "Clear Cache"),
      selectInput("chirts_var", "CHIRTS Variable", choices = c("Tmax", "Tmin", "RHum", "HeatIndex", "All"), selected = "All"),
      actionButton("open_tempdir", "Open Temp Folder") # button to open the temp folder
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Log", verbatimTextOutput("log")),
        tabPanel("Progress", verbatimTextOutput("progress_text")),
        tabPanel("Summary Plot", plotOutput("summary_plot"))
      )
    )
  )
)

# Server logic for the Shiny app
server <- function(input, output, session) {
  rv <- reactiveValues(
    files = NULL,
    monthly_totals = NULL,
    data_type = NULL,
    progress_msg = "",
    coord_choices = NULL
  )
  plan(multisession)
  
  output$progress_text <- renderText({
    rv$progress_msg
  })
  
  # Update coordinate dropdown after processing
  observe({
    if (!is.null(rv$monthly_totals) && "Name" %in% names(rv$monthly_totals)) {
      choices <- unique(rv$monthly_totals$Name)
      choices <- c("All", choices) # Add "All" option
      rv$coord_choices <- choices
      output$coord_selector <- renderUI({
        selectInput("selected_coord", "Select Coordinate for Plot", choices = choices, selected = "All")
      })
    } else {
      output$coord_selector <- renderUI({ NULL })
    }
  })
  
  observeEvent(input$run, {
    req(input$coord_file)
    msg <- function(text) {
      print(text)
      rv$progress_msg <<- paste0(rv$progress_msg, text, "\n")
    }
    rv$progress_msg <- ""
    msg("Validating input file...")
    coord_file <- input$coord_file$datapath
    dates <- c(as.character(input$date_range[1]), as.character(input$date_range[2]))
    msg("Reading and validating coordinates and dates...")
    coord <- validate_inputs(coord_file, dates)
    unique_coords <- unique(coord[, c("Longitude", "Latitude", "Name")])
    all_files <- list()
    all_monthly_totals <- list()
    all_downloaded_data <- vector("list", nrow(unique_coords))
    names(all_downloaded_data) <- unique_coords$Name
    
    msg(paste0("Starting download for ", nrow(unique_coords), " locations..."))
    # 1. Download all data for all coordinates first
    for (i in seq_len(nrow(unique_coords))) {
      name <- unique_coords$Name[i]
      msg(paste0("[", i, "/", nrow(unique_coords), "] Downloading data for ", name, "..."))
      lonlat <- unique_coords[i, ]
      lonlat <- data.frame(lon = lonlat$Longitude, lat = lonlat$Latitude)
      if (input$data_type == "CHIRPS") {
        data <- download_chirps_data(lonlat, dates, progress = msg)
      } else {
        data <- download_chirts_data(lonlat, dates, var = input$chirts_var, progress = msg)
      }
      if (is.null(data)) {
        msg(paste0("[", i, "] Failed to download data for ", name, ". Skipping."))
        all_downloaded_data[[name]] <- NULL
        next
      }
      all_downloaded_data[[name]] <- data
      msg(paste0("[", i, "] Download complete for ", name, "."))
    }
    
    msg("All downloads complete. Starting processing and report generation...")
    # 2. After all downloads, process and generate reports
    for (i in seq_len(nrow(unique_coords))) {
      name <- unique_coords$Name[i]
      data <- all_downloaded_data[[name]]
      if (is.null(data)) {
        msg(paste0("[", i, "] No data for ", name, ". Skipping processing."))
        next
      }
      msg(paste0("[", i, "] Processing data for ", name, "..."))
      daily_data <- data
      processed_data <- process_climate_data(data, input$data_type, progress = msg)
      msg(paste0("[", i, "] Generating report for ", name, "..."))
      reports <- generate_report(processed_data$data, coord, input$data_type, progress = msg)
      processed_data$monthly_totals$Name <- name
      all_monthly_totals[[i]] <- processed_data$monthly_totals
      safe_name <- make.names(name)
      output_file <- tempfile(fileext = paste0("_", safe_name, ".", input$output_format))
      msg(paste0("[", i, "] Writing output files for ", name, "..."))
      if (input$output_format == "xlsx") {
        write_xlsx(list(
          Daily_Data = daily_data,
          Data = processed_data$data, 
          Monthly_Totals = processed_data$monthly_totals, 
          Yearly_Totals = processed_data$yearly_totals,
          Monthly_Report = reports$monthly,
          Yearly_Report = reports$yearly
        ), output_file)
      } else {
        write.csv(daily_data, sub("\\.csv$", "_daily.csv", output_file), row.names = FALSE)
        write.csv(processed_data$data, sub("\\.csv$", "_data.csv", output_file), row.names = FALSE)
        write.csv(processed_data$monthly_totals, sub("\\.csv$", "_monthly.csv", output_file), row.names = FALSE)
        write.csv(processed_data$yearly_totals, sub("\\.csv$", "_yearly.csv", output_file), row.names = FALSE)
        write.csv(reports$monthly, sub("\\.csv$", "_monthly_report.csv", output_file), row.names = FALSE)
        write.csv(reports$yearly, sub("\\.csv$", "_yearly_report.csv", output_file), row.names = FALSE)
      }
      all_files[[paste0("file_", i)]] <- output_file
      msg(paste0("[", i, "] Done with ", name, "."))
    }
    if (length(all_monthly_totals) > 0) {
      msg("Combining all monthly totals...")
      rv$monthly_totals <- dplyr::bind_rows(all_monthly_totals)
    }
    rv$files <- unlist(all_files)
    rv$data_type <- input$data_type
    msg("All data processing completed.")
    showNotification("All data processing completed", type = "message")
    beep()
    output$log <- renderText("Process completed. Check the log for details.")
  })
  
  output$summary_plot <- renderPlot({
    req(rv$monthly_totals)
    df <- rv$monthly_totals
    # If not "All", filter by selected coordinate
    if (!is.null(input$selected_coord) && input$selected_coord != "All" && "Name" %in% names(df)) {
      df <- df[df$Name == input$selected_coord, ]
    }
    if (rv$data_type == "CHIRPS") {
      ggplot(df, aes(x = as.Date(paste(year, month, "01", sep = "-")), 
                     y = monthly_total_precip, color = Name)) +
        geom_line() +
        labs(x = "Date", y = "Monthly Precipitation (mm)",
             title = ifelse(is.null(input$selected_coord) || input$selected_coord == "All",
                            "Monthly Precipitation for All Coordinates",
                            paste("Monthly Precipitation for", input$selected_coord))) +
        theme_minimal()
    } else {
      if (!is.null(input$selected_coord) && input$selected_coord == "All") {
        # All coordinates, each variable for each Name
        df_long <- tidyr::pivot_longer(
          df,
          cols = c(monthly_mean_tmax, monthly_mean_tmin, monthly_mean_rhum, monthly_mean_heatindex),
          names_to = "Variable",
          values_to = "Value"
        )
        ggplot(df_long, aes(x = as.Date(paste(year, month, "01", sep = "-")), y = Value, color = interaction(Name, Variable))) +
          geom_line() +
          labs(x = "Date", y = "Monthly Mean Values",
               title = "Monthly Mean Tmax, Tmin, RHum, and HeatIndex for All Coordinates") +
          theme_minimal()
      } else {
        # Single coordinate, colored by variable
        ggplot(df) +
          geom_line(aes(x = as.Date(paste(year, month, "01", sep = "-")), 
                        y = monthly_mean_tmax, color = "Tmax")) +
          geom_line(aes(x = as.Date(paste(year, month, "01", sep = "-")), 
                        y = monthly_mean_tmin, color = "Tmin")) +
          geom_line(aes(x = as.Date(paste(year, month, "01", sep = "-")), 
                        y = monthly_mean_rhum, color = "RHum")) +
          geom_line(aes(x = as.Date(paste(year, month, "01", sep = "-")), 
                        y = monthly_mean_heatindex, color = "HeatIndex")) +
          scale_color_manual(values = c("Tmax" = "red", "Tmin" = "blue", "RHum" = "green", "HeatIndex" = "purple")) +
          labs(x = "Date", y = "Monthly Mean Values", title = paste("Monthly Mean Tmax, Tmin, RHum, and HeatIndex for", input$selected_coord)) +
          theme_minimal()
      }
    }
  })
  
  output$download_data_files <- downloadHandler(
    filename = function() {
      paste0("Climate_data_", Sys.Date(), ".zip")
    },
    content = function(file) {
      req(rv$files)
      zip::zipr(file, rv$files)
    },
    contentType = "application/zip"
  )
  
  observeEvent(input$save_config, {
    config <- list(
      data_type = input$data_type,
      date_range = as.character(input$date_range),
      save_raw_data = input$save_raw_data,
      output_format = input$output_format
    )
    config_file <- tempfile(fileext = ".json")
    write_json(config, config_file)
    showModal(modalDialog(
      title = "Save Config",
      downloadButton("download_config", "Download Config"),
      easyClose = TRUE,
      footer = NULL
    ))
    output$download_config <- downloadHandler(
      filename = function() {
        "config.json"
      },
      content = function(file) {
        file.copy(config_file, file)
      },
      contentType = "application/json"
    )
  })
  
  observeEvent(input$config_file, {
    config_file <- input$config_file$datapath
    if (!is.null(config_file)) {
      config <- fromJSON(config_file)
      updateSelectInput(session, "data_type", selected = config$data_type)
      updateDateRangeInput(session, "date_range", start = config$date_range[1], end = config$date_range[2])
      updateCheckboxInput(session, "save_raw_data", value = config$save_raw_data)
      updateSelectInput(session, "output_format", selected = config$output_format)
    }
  })
  
  # Add UI and server logic for clearing cache
  download_cache_dirs <- c("chirps_cache", "chirts_cache")
  
  # In server: handle cache clearing
  observeEvent(input$clear_cache, {
    for (cache_dir in download_cache_dirs) {
      if (dir.exists(cache_dir)) {
        unlink(cache_dir, recursive = TRUE, force = TRUE)
        dir.create(cache_dir, showWarnings = FALSE)
      }
    }
    showNotification("Cache cleared.", type = "message")
    rv$progress_msg <- "Cache cleared."
  })
  
  # Add handler to open tempdir in file explorer
  observeEvent(input$open_tempdir, {
    temp_path <- normalizePath(tempdir(), winslash = "\\", mustWork = TRUE)
    if (file.exists(temp_path)) {
      tryCatch({
        shell.exec(temp_path)
      }, error = function(e) {
        showNotification(paste("Failed to open temp folder:", e$message), type = "error")
      })
    } else {
      showNotification("Temp folder does not exist.", type = "error")
    }
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)