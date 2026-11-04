# 원본 R 코드(Diercks, Katz & Wright 공개 저장소)를 그대로 불러와 실행하는 검증용 스크립트.
# 사용: Rscript tests/run_dkw_r.R <원본저장소경로> <입력csv> <분포출력csv> <통계량출력csv> <strike_int> <horizon> <moment_adj>
# get_moments의 첨도(DescTools::Kurt)만 빼고 실행해. 연구에 쓰지 않는 통계량이고 패키지가 없는 환경이 있어서야.

args <- commandArgs(trailingOnly = TRUE)
repo <- args[1]
suppressMessages(source(file.path(repo, "code/convert_trades_to_pdfs/convert_trade_level_data_cdfs.R")))

get_moments <- function(df, moment_adjustment) {
  df %>%
    group_by(date, contract_preamble, expiry_date) %>%
    summarise(
      daily_volume = sum(daily_volume),
      mean     = sum(probability * strike, na.rm = TRUE) / sum(probability, na.rm = TRUE) + moment_adjustment,
      median   = weightedMedian(strike, w = probability, na.rm = TRUE, interpolate = FALSE) + moment_adjustment,
      mode     = fmode(strike, w = probability, na.rm = TRUE, ties = 'first') + moment_adjustment,
      skewness = weightedGMSkew(strike, w = probability, na.rm = TRUE),
      variance = sum(probability * (strike - (sum(probability * strike) / sum(probability)))^2, na.rm = TRUE) / sum(probability, na.rm = TRUE),
      .groups = "drop"
    )
}

invisible(capture.output(suppressMessages(
  extract_distributions(input_file = args[2],
                        output_distributions = args[3],
                        output_moments = args[4],
                        output_wide = tempfile(fileext = ".csv"),
                        strike_int = as.numeric(args[5]),
                        days_before_horizon = as.integer(args[6]),
                        moment_adjustment = as.numeric(args[7]))
)))
cat("R done\n")
