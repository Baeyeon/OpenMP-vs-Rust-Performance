mod mod_parent {
    include!("mod.rs");
}

fn main() {
    mod_parent::run_all_benchmarks();
}
