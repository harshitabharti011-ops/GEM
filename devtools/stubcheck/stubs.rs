// Minimal stand-ins for GEM's external crates, so src/aigpdk.rs, src/macros.rs
// and src/aig.rs can be type- and borrow-checked with plain rustc, offline and
// without the workspace's dependency tree.
//
// This is a STOPGAP for machines with no cargo registry access. `cargo check`
// and `cargo test` remain authoritative -- these stubs only have to be
// API-shaped, not correct.
#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
pub type CompactString = String;

pub mod sverilogparse { #[derive(Debug,Clone,Copy)] pub struct SVerilogRange(pub isize, pub isize); }

pub mod netlistdb {
    use super::CompactString;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Direction { I, O }
    pub trait GeneralPinName { fn dbg_fmt_pin(&self) -> String; }
    pub trait GeneralHierName { fn dbg_fmt_hier(&self) -> String; }
    #[derive(Clone)] pub struct HierName(pub String);
    impl HierName { pub fn single(s: CompactString) -> Self { HierName(s) } }
    impl GeneralHierName for HierName { fn dbg_fmt_hier(&self) -> String { self.0.clone() } }
    pub type PinName = (HierName, CompactString, Option<isize>);
    impl GeneralPinName for PinName {
        fn dbg_fmt_pin(&self) -> String { format!("{}.{}", self.0.0, self.1) } }
    impl GeneralPinName for (HierName, &CompactString, Option<isize>) {
        fn dbg_fmt_pin(&self) -> String { format!("{}.{}", self.0.0, self.1) } }
    pub struct Csr { pub start: Vec<usize>, pub items: Vec<usize> }
    impl Csr { pub fn iter_set(&self, i: usize) -> std::vec::IntoIter<usize> {
        self.items[self.start[i]..self.start[i+1]].to_vec().into_iter() } }
    pub struct NetlistDB {
        pub num_pins: usize, pub num_cells: usize,
        pub pin2net: Vec<usize>, pub pin2cell: Vec<usize>,
        pub celltypes: Vec<CompactString>, pub cellnames: Vec<HierName>,
        pub pinnames: Vec<PinName>, pub pindirect: Vec<Direction>,
        pub net2pin: Csr, pub cell2pin: Csr,
        pub net_zero: Option<usize>, pub net_one: Option<usize>,
    }
    pub trait LeafPinProvider {
        fn direction_of(&self, m:&CompactString, p:&CompactString, i:Option<isize>) -> Direction;
        fn width_of(&self, m:&CompactString, p:&CompactString) -> Option<super::sverilogparse::SVerilogRange>;
    }
}

pub mod indexmap {
    use std::ops::Index;
    // NB: the real IndexMap implements Default with no K/V bounds. Deriving it
    // here would add `V: Default` and wrongly reject AIG's derive(Default).
    #[derive(Debug, Clone)] pub struct IndexMap<K, V>(pub Vec<(K, V)>);
    impl<K, V> Default for IndexMap<K, V> { fn default() -> Self { IndexMap(Vec::new()) } }
    impl<K: PartialEq + Copy, V> IndexMap<K, V> {
        pub fn len(&self) -> usize { self.0.len() }
        pub fn get(&self, k: &K) -> Option<&V> { self.0.iter().find(|e| e.0 == *k).map(|e| &e.1) }
        pub fn get_mut(&mut self, k: &K) -> Option<&mut V> { self.0.iter_mut().find(|e| e.0 == *k).map(|e| &mut e.1) }
        pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> { self.0.iter().map(|e| (&e.0, &e.1)) }
        pub fn insert(&mut self, k: K, v: V) { self.0.push((k, v)); }
        pub fn entry(&mut self, k: K) -> Entry<'_, K, V> { Entry { m: self, k } }
    }
    pub struct Entry<'a, K, V> { m: &'a mut IndexMap<K, V>, k: K }
    impl<'a, K: PartialEq + Copy, V: Default> Entry<'a, K, V> {
        pub fn or_default(self) -> &'a mut V { self.or_insert_with(V::default) } }
    impl<'a, K: PartialEq + Copy, V> Entry<'a, K, V> {
        pub fn or_insert(self, v: V) -> &'a mut V { self.or_insert_with(|| v) }
        pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
            if let Some(i) = self.m.0.iter().position(|e| e.0 == self.k) { return &mut self.m.0[i].1 }
            self.m.0.push((self.k, f())); let n = self.m.0.len(); &mut self.m.0[n-1].1 } }
    impl<K: PartialEq + Copy, V> Index<usize> for IndexMap<K, V> {
        type Output = V; fn index(&self, i: usize) -> &V { &self.0[i].1 } }
    impl<K: PartialEq + Copy, V> Index<&K> for IndexMap<K, V> {
        type Output = V; fn index(&self, k: &K) -> &V { self.get(k).unwrap() } }
    impl<'a, K, V> IntoIterator for &'a IndexMap<K, V> {
        type Item = (&'a K, &'a V);
        type IntoIter = std::iter::Map<std::slice::Iter<'a,(K,V)>, fn(&'a (K,V)) -> (&'a K, &'a V)>;
        fn into_iter(self) -> Self::IntoIter { self.0.iter().map(|e| (&e.0, &e.1)) } }

    #[derive(Debug, Clone)] pub struct IndexSet<T>(pub Vec<T>);
    impl<T> Default for IndexSet<T> { fn default() -> Self { IndexSet(Vec::new()) } }
    impl<T: PartialEq + Copy> IndexSet<T> {
        pub fn new() -> Self { IndexSet(Vec::new()) }
        pub fn len(&self) -> usize { self.0.len() }
        pub fn insert(&mut self, v: T) -> bool { if self.0.contains(&v) { false } else { self.0.push(v); true } }
        pub fn contains(&self, v: &T) -> bool { self.0.contains(v) }
        pub fn get_index(&self, i: usize) -> Option<&T> { self.0.get(i) }
        pub fn is_empty(&self) -> bool { self.0.is_empty() }
        pub fn swap_remove(&mut self, v: &T) -> bool {
            match self.0.iter().position(|x| x == v) {
                Some(i) => { self.0.swap_remove(i); true },
                None => false,
            } }
        pub fn iter(&self) -> std::slice::Iter<'_, T> { self.0.iter() } }
    impl<T: PartialEq + Copy> FromIterator<T> for IndexSet<T> {
        fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Self {
            let mut s = IndexSet::new();
            for v in it { s.insert(v); }
            s } }
    impl<T> IntoIterator for IndexSet<T> {
        type Item = T; type IntoIter = std::vec::IntoIter<T>;
        fn into_iter(self) -> Self::IntoIter { self.0.into_iter() } }
    impl<'a, T> IntoIterator for &'a IndexSet<T> {
        type Item = &'a T; type IntoIter = std::slice::Iter<'a, T>;
        fn into_iter(self) -> Self::IntoIter { self.0.iter() } }
}
