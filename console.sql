select count(*) from benchmark b;

select avg(bm.avg_best_cost), avg(bm.avg_execution_time) from (select * from benchmark b order by b.avg_execution_time limit 100) as bm where bm.selection_operator = 'RouletteSelection';

select count(*) from (select * from benchmark b order by b.avg_execution_time limit 100) as bm where bm.selection_operator = 'RouletteSelection';

select * from benchmark bm where bm.population_size = 50 order by bm.avg_best_cost limit 1