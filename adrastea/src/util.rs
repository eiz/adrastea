/*
 * This file is part of Adrastea.
 *
 * Adrastea is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License as published by the Free Software
 * Foundation, version 3.
 *
 * Adrastea is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along
 * with Adrastea. If not, see <https://www.gnu.org/licenses/>.
 */

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

pub fn ceil_div(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}
