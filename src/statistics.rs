use serde::Serialize;

#[derive(Default, Serialize)]
pub struct Statistics {
    pub sierra_type_count: usize,
    pub sierra_libfunc_count: usize,
    pub sierra_statement_count: usize,
    pub sierra_func_count: usize,
}

impl Statistics {
    pub fn builder() -> StatisticsBuilder {
        StatisticsBuilder::default()
    }
}

#[derive(Default)]
pub struct StatisticsBuilder {
    pub sierra_type_count: Option<usize>,
    pub sierra_libfunc_count: Option<usize>,
    pub sierra_statement_count: Option<usize>,
    pub sierra_func_count: Option<usize>,
}

impl StatisticsBuilder {
    pub fn build(self) -> Option<Statistics> {
        Some(Statistics {
            sierra_type_count: self.sierra_type_count?,
            sierra_libfunc_count: self.sierra_libfunc_count?,
            sierra_statement_count: self.sierra_statement_count?,
            sierra_func_count: self.sierra_func_count?,
        })
    }
}
