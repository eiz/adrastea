pub struct ElidingRangeIterator {
    end: usize,
    current: usize,
    threshold: usize,
    edge_items: usize,
}

impl ElidingRangeIterator {
    pub fn new(n: usize, elide_threshold: usize, edge_items: usize) -> Self {
        Self { end: n, current: 0, threshold: elide_threshold, edge_items }
    }
}

impl Iterator for ElidingRangeIterator {
    type Item = (bool, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let result = self.current;
            self.current += 1;

            if self.end > self.threshold {
                if self.current >= self.edge_items && self.current < self.end - self.edge_items {
                    self.current = self.end - self.edge_items;
                    return Some((true, result));
                }
            }

            Some((false, result))
        } else {
            None
        }
    }
}
